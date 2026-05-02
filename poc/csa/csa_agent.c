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

    if (a->st == ST_IDLE || sess > a->sess) {
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
        if (a->no_revert) {
            log_msg("state=COMMITTED (no-revert mode)");
            to_idle(a);
            return;
        }
        a->st = ST_VERIFY;
        a->t_revert_ms = now_ms() + a->t_revert_ms;
        log_msg("state=VERIFY revert_at=+%lldms", a->t_revert_ms - now_ms());
    } else if (a->st == ST_VERIFY && now >= a->t_revert_ms) {
        log_msg("REVERT triggered (no traffic seen on new channel)");
        run_iw(a->iface, a->prev_chan, a->prev_ht);
        to_idle(a);
    }
}

int main(int argc, char **argv) {
    int no_revert = 0;
    int argi = 1;
    if (argi < argc && strcmp(argv[argi], "--no-revert") == 0) {
        no_revert = 1; argi++;
    }
    if (argc - argi < 2) {
        fprintf(stderr, "usage: %s [--no-revert] <port> <iface>\n", argv[0]);
        return 2;
    }
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
    log_msg("csa_agent listening port=%d iface=%s no_revert=%d",
            port, iface, no_revert);

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
