/* csa_agent: one-way Channel Switch Announcement receiver.
 *
 * Binds UDP <port> on the vehicle (replaces link_controller's 5801 binding
 * during the bench test). Parses csa_commit JSON frames, schedules a single
 * `iw set channel` at the target monotonic deadline, and reverts if no UDP
 * traffic is observed within t_revert_ms after the switch.
 *
 * Build:  arm-linux-gnueabihf-gcc -static -Os -o csa_agent.armhf csa_agent.c
 *
 * The JSON parser is intentionally minimal: it scans for "key":<value> pairs
 * with single-token values (string/int). Robust enough for our line-delimited
 * line-per-packet format, not a general parser.
 */
#define _POSIX_C_SOURCE 200809L
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static long long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL;
}

static void log_msg(const char *fmt, ...) {
    char tbuf[64];
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm; localtime_r(&ts.tv_sec, &tm);
    strftime(tbuf, sizeof(tbuf), "%H:%M:%S", &tm);
    fprintf(stderr, "[%s.%03ld] ", tbuf, ts.tv_nsec / 1000000L);
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr);
}

/* ----- minimal JSON field extractor ----- */
static const char *find_key(const char *buf, const char *key) {
    size_t klen = strlen(key);
    const char *p = buf;
    while ((p = strstr(p, key))) {
        if (p > buf && p[-1] == '"' && p[klen] == '"') {
            const char *q = p + klen + 1;
            while (*q == ' ') q++;
            if (*q == ':') return q + 1;
        }
        p += klen;
    }
    return NULL;
}

static bool get_str(const char *buf, const char *key, char *out, size_t outlen) {
    const char *p = find_key(buf, key);
    if (!p) return false;
    while (*p == ' ') p++;
    if (*p != '"') return false;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i + 1 < outlen) out[i++] = *p++;
    out[i] = 0;
    return true;
}

static bool get_int(const char *buf, const char *key, long long *out) {
    const char *p = find_key(buf, key);
    if (!p) return false;
    while (*p == ' ') p++;
    char *end;
    long long v = strtoll(p, &end, 10);
    if (end == p) return false;
    *out = v;
    return true;
}

/* ----- allowlist + DFS guard + cooldown ----- */
/* 5 GHz DFS channels (UNII-2 + UNII-2-extended). Hopping into these without
 * CAC is a regulatory violation in most regions, so they are blocked unless
 * --allow-dfs is set on the command line. */
static const int DFS_CHANS[] = {
    52, 56, 60, 64,
    100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144,
};

static bool is_dfs(int chan) {
    for (size_t i = 0; i < sizeof(DFS_CHANS)/sizeof(DFS_CHANS[0]); i++) {
        if (DFS_CHANS[i] == chan) return true;
    }
    return false;
}

#define MAX_ALLOW 32
typedef struct { int chan; char ht[8]; } allow_t;
static allow_t g_allow[MAX_ALLOW];
static int g_allow_n = 0;
static int g_allow_dfs = 0;
static long long g_cooldown_ms = 2000;       /* 0 disables */
static long long g_last_switch_ms = -1;      /* -1 = no prior switch */

/* "149/HT20,153/HT20,161/HT40+" -> g_allow[]. Returns 0 on success, -1 on parse error. */
static int parse_allowlist(const char *s) {
    g_allow_n = 0;
    const char *p = s;
    while (*p && g_allow_n < MAX_ALLOW) {
        char *end;
        long c = strtol(p, &end, 10);
        if (end == p || *end != '/') return -1;
        p = end + 1;
        const char *htstart = p;
        while (*p && *p != ',') p++;
        size_t htlen = (size_t)(p - htstart);
        if (htlen == 0 || htlen >= sizeof(g_allow[0].ht)) return -1;
        g_allow[g_allow_n].chan = (int)c;
        memcpy(g_allow[g_allow_n].ht, htstart, htlen);
        g_allow[g_allow_n].ht[htlen] = 0;
        g_allow_n++;
        if (*p == ',') p++;
    }
    return (*p == 0) ? 0 : -1;
}

static bool allowlist_ok(int chan, const char *ht) {
    if (g_allow_n == 0) return true;  /* unset == permissive */
    for (int i = 0; i < g_allow_n; i++) {
        if (g_allow[i].chan == chan && strcmp(g_allow[i].ht, ht) == 0)
            return true;
    }
    return false;
}

/* ----- iw spawning ----- */
static int run_iw(const char *iface, int chan, const char *ht) {
    char chan_s[16]; snprintf(chan_s, sizeof(chan_s), "%d", chan);
    log_msg("iw dev %s set channel %s %s", iface, chan_s, ht);
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        execl("/usr/sbin/iw", "iw", "dev", iface, "set", "channel",
              chan_s, ht, (char*)NULL);
        execlp("iw", "iw", "dev", iface, "set", "channel",
               chan_s, ht, (char*)NULL);
        _exit(127);
    }
    int st;
    waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

/* ----- state machine ----- */
typedef enum { ST_IDLE, ST_ARMED, ST_VERIFY } state_t;

typedef struct {
    state_t st;
    long long sess;
    long long t_switch_ms;       /* monotonic */
    long long t_revert_ms;       /* monotonic */
    int target_chan, prev_chan;
    char target_ht[8], prev_ht[8];
    char iface[16];
    int no_revert;               /* primary role: switch and stay */
} agent_t;

static void to_idle(agent_t *a) {
    a->st = ST_IDLE;
    log_msg("state=IDLE");
}

static void on_commit(agent_t *a, const char *buf, ssize_t len) {
    (void)len;
    long long sess, seq, dt, t_revert, target_chan, prev_chan;
    char target_ht[8] = "HT20", prev_ht[8] = "HT20";
    if (!get_int(buf, "sess", &sess)) return;
    if (!get_int(buf, "seq", &seq)) seq = 0;
    if (!get_int(buf, "dt_to_switch_ms", &dt)) return;
    if (!get_int(buf, "t_revert_ms", &t_revert)) t_revert = 3000;
    if (!get_int(buf, "target_chan", &target_chan)) return;
    if (!get_int(buf, "prev_chan", &prev_chan)) prev_chan = 0;
    get_str(buf, "target_ht", target_ht, sizeof(target_ht));
    get_str(buf, "prev_ht", prev_ht, sizeof(prev_ht));

    long long now = now_ms();
    long long t_switch = now + dt;

    /* Defense-in-depth target validation. Applied to every csa_commit so that
     * even refresh frames carrying a tampered target are caught. The range
     * check runs first so that DFS / allowlist comparisons cannot be tricked
     * by a value that wraps under (int) truncation. */
    if (target_chan < 1 || target_chan > 200) {
        log_msg("REJECT sess=%lld seq=%lld: target_chan=%lld out of range",
                sess, seq, target_chan);
        return;
    }
    if (!g_allow_dfs && is_dfs((int)target_chan)) {
        log_msg("REJECT sess=%lld seq=%lld: target ch%lld is DFS (use --allow-dfs)",
                sess, seq, target_chan);
        return;
    }
    if (!allowlist_ok((int)target_chan, target_ht)) {
        log_msg("REJECT sess=%lld seq=%lld: target ch%lld %s not in allowlist",
                sess, seq, target_chan, target_ht);
        return;
    }

    /* Tail-of-burst csa_commit frames arriving after SWITCH on the new
     * channel confirm the link the same way other UDP traffic would.
     * PROTOCOL.md says "any UDP frame (CSA or stats) before deadline"
     * confirms; without this branch a same-sess refresh would be silently
     * dropped and we'd revert despite the channel being healthy. */
    if (a->st == ST_VERIFY && sess == a->sess) {
        log_msg("VERIFY heartbeat: csa_commit seq=%lld -> COMMITTED", seq);
        to_idle(a);
        return;
    }

    if (a->st == ST_IDLE || sess > a->sess) {
        /* Cooldown only gates NEW sessions; same-session refreshes are always
         * allowed (they refine T_switch within ±20ms of the original). */
        if (g_cooldown_ms > 0 && g_last_switch_ms >= 0) {
            long long since = now - g_last_switch_ms;
            if (since < g_cooldown_ms) {
                log_msg("REJECT sess=%lld seq=%lld: cooldown %lldms remaining",
                        sess, seq, g_cooldown_ms - since);
                return;
            }
        }
        a->st = ST_ARMED;
        a->sess = sess;
        a->t_switch_ms = t_switch;
        a->t_revert_ms = t_revert;
        a->target_chan = (int)target_chan;
        a->prev_chan = (int)prev_chan;
        snprintf(a->target_ht, sizeof(a->target_ht), "%s", target_ht);
        snprintf(a->prev_ht, sizeof(a->prev_ht), "%s", prev_ht);
        log_msg("ARMED sess=%lld seq=%lld dt=%lldms target=ch%d %s prev=ch%d %s revert=%lldms",
                sess, seq, dt, a->target_chan, a->target_ht,
                a->prev_chan, a->prev_ht, t_revert);
        return;
    }

    if (a->st == ST_ARMED && sess == a->sess) {
        long long delta = t_switch - a->t_switch_ms;
        if (delta < 0) delta = -delta;
        if (delta <= 20) {
            log_msg("REFRESH seq=%lld dt=%lldms (drift=%lldms ok)", seq, dt, t_switch - a->t_switch_ms);
        } else {
            log_msg("REFRESH seq=%lld dt=%lldms (drift=%lldms exceeds ±20ms, ignored)",
                    seq, dt, t_switch - a->t_switch_ms);
        }
    }
}

static void tick(agent_t *a) {
    long long now = now_ms();
    if (a->st == ST_ARMED && now >= a->t_switch_ms) {
        log_msg("SWITCH at +%lldms (target ch%d %s)",
                now - a->t_switch_ms, a->target_chan, a->target_ht);
        run_iw(a->iface, a->target_chan, a->target_ht);
        /* Anchor cooldown at the scheduled switch time, not after iw(8)
         * returns, so the gap is deterministic regardless of how long the
         * iw spawn takes (typ. 50–300 ms). */
        g_last_switch_ms = now;
        if (a->no_revert) {
            log_msg("state=COMMITTED (no-revert mode)");
            to_idle(a);
            return;
        }
        a->st = ST_VERIFY;
        /* Revert deadline is intentionally measured from after iw returns:
         * we want to wait t_revert_ms of silence on the *new* channel. */
        a->t_revert_ms = now_ms() + a->t_revert_ms;
        log_msg("state=VERIFY revert_at=+%lldms", a->t_revert_ms - now_ms());
    } else if (a->st == ST_VERIFY && now >= a->t_revert_ms) {
        log_msg("REVERT triggered (no traffic seen on new channel)");
        run_iw(a->iface, a->prev_chan, a->prev_ht);
        g_last_switch_ms = now;  /* revert is itself a hop */
        to_idle(a);
    }
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s [--no-revert] [--allow-dfs] [--allowlist CH/HT,...] "
        "[--cooldown-ms N] <port> <iface>\n"
        "  --allowlist  comma-separated CH/HT pairs accepted as targets\n"
        "               (e.g. 149/HT20,153/HT20,161/HT40+); empty = permissive\n"
        "  --allow-dfs  permit hops into 5GHz DFS channels (52..144)\n"
        "  --cooldown-ms minimum gap between channel changes (default 2000, 0 disables)\n",
        argv0);
}

int main(int argc, char **argv) {
    int no_revert = 0;
    const char *allowlist_arg = NULL;
    int argi = 1;
    while (argi < argc && argv[argi][0] == '-' && argv[argi][1] == '-') {
        const char *flag = argv[argi];
        if (strcmp(flag, "--no-revert") == 0) {
            no_revert = 1; argi++;
        } else if (strcmp(flag, "--allow-dfs") == 0) {
            g_allow_dfs = 1; argi++;
        } else if (strcmp(flag, "--allowlist") == 0 && argi + 1 < argc) {
            allowlist_arg = argv[argi + 1]; argi += 2;
        } else if (strcmp(flag, "--cooldown-ms") == 0 && argi + 1 < argc) {
            g_cooldown_ms = strtoll(argv[argi + 1], NULL, 10); argi += 2;
        } else {
            fprintf(stderr, "unknown flag: %s\n", flag);
            usage(argv[0]);
            return 2;
        }
    }
    if (allowlist_arg && parse_allowlist(allowlist_arg) < 0) {
        fprintf(stderr, "bad --allowlist: %s\n", allowlist_arg);
        return 2;
    }
    /* Warn (don't fail) on a configuration that's effectively unreachable:
     * an allowlist entry sitting on a DFS channel will always be killed by
     * the DFS guard unless --allow-dfs is also set. Easy footgun otherwise. */
    if (g_allow_n > 0 && !g_allow_dfs) {
        for (int i = 0; i < g_allow_n; i++) {
            if (is_dfs(g_allow[i].chan)) {
                fprintf(stderr,
                    "warning: allowlist entry %d/%s is a DFS channel and "
                    "will always be rejected by the DFS guard. "
                    "Pass --allow-dfs to enable it.\n",
                    g_allow[i].chan, g_allow[i].ht);
            }
        }
    }
    if (argc - argi < 2) { usage(argv[0]); return 2; }
    int port = atoi(argv[argi++]);
    const char *iface = argv[argi++];

    int s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons(port),
                              .sin_addr.s_addr = htonl(INADDR_ANY) };
    if (bind(s, (struct sockaddr*)&a, sizeof(a)) < 0) { perror("bind"); return 1; }

    agent_t ag = { .st = ST_IDLE, .no_revert = no_revert };
    snprintf(ag.iface, sizeof(ag.iface), "%s", iface);
    log_msg("csa_agent listening port=%d iface=%s no_revert=%d "
            "allow_dfs=%d cooldown_ms=%lld allowlist=%d entries",
            port, iface, no_revert, g_allow_dfs, g_cooldown_ms, g_allow_n);

    char buf[4096];
    struct timeval tv;
    fd_set rfds;
    while (1) {
        FD_ZERO(&rfds); FD_SET(s, &rfds);
        tv.tv_sec = 0; tv.tv_usec = 5000;  /* 5 ms tick */
        int rv = select(s + 1, &rfds, NULL, NULL, &tv);
        if (rv < 0 && errno == EINTR) continue;
        if (rv > 0 && FD_ISSET(s, &rfds)) {
            ssize_t n = recv(s, buf, sizeof(buf) - 1, 0);
            if (n > 0) {
                buf[n] = 0;
                char type[32] = "";
                get_str(buf, "type", type, sizeof(type));
                if (strncmp(type, "csa_commit", 10) == 0) {
                    on_commit(&ag, buf, n);
                } else if (ag.st == ST_VERIFY) {
                    log_msg("VERIFY heartbeat: type=%s -> COMMITTED", type);
                    to_idle(&ag);  /* MVP: any traffic after switch confirms */
                }
            }
        }
        tick(&ag);
    }
    return 0;
}
