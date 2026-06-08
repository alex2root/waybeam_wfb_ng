/* tier1_trend.h -- causal trend-feature ring buffer for the Tier-1 model.
 *
 * The trend-enabled model (train_tier1.py --trend) consumes 11 inputs:
 *   reduced base (5): rssi, pkt_all, pkt_lost, pkt_fec_recovered, pkt_uniq
 *   trend       (6): rssi_slope, rssi_mean, loss_rate, fec_rate,
 *                    time_since_loss, dropout_frac
 * m2cgen's tier1_model.c only emits score(double in[11], double out[3]) -- it
 * does NOT compute the trend columns. This header is the missing half: a tiny
 * fixed-size ring buffer that turns the per-record reduced base into the full
 * 11-feature vector, bit-for-bit matching wfb_schema.TrendState (so the trained
 * tree sees on the SSC338Q exactly what it saw in training).
 *
 * Usage on the air unit (one t1_trend per link):
 *     t1_trend tr; t1_trend_init(&tr);
 *     // each wfb_rx -Y interval: reduce antennas to best-chain RSSI yourself
 *     // (RSSI sentinel TIER1_SIGNAL_LOST_RSSI when no antenna reported), then:
 *     double feat[11];
 *     t1_trend_push(&tr, rssi, pkt_all, pkt_lost, pkt_fec_recovered, pkt_uniq, feat);
 *     double out[3]; score(feat, out);   // argmax -> 0 ok / 1 degraded / 2 critical
 *
 * Dependency-free C89-ish; no malloc, no libm (only + - * and divide).
 */
#ifndef TIER1_TREND_H
#define TIER1_TREND_H

#ifndef TIER1_TREND_WINDOW
#define TIER1_TREND_WINDOW 10        /* must equal the trainer's --window */
#endif
#define TIER1_TREND_SINCE_CAP 200.0  /* matches TrendState since_cap */
#define TIER1_SIGNAL_LOST_RSSI (-128.0)

typedef struct {
    double rssi[TIER1_TREND_WINDOW];
    double lost[TIER1_TREND_WINDOW];
    double uniq[TIER1_TREND_WINDOW];
    double fec[TIER1_TREND_WINDOW];
    double drop[TIER1_TREND_WINDOW];
    int    head;        /* next write slot */
    int    n;           /* fill count (<= WINDOW) */
    double since_loss;  /* records since last loss/dropout, capped */
} t1_trend;

static void t1_trend_init(t1_trend *t) {
    t->head = 0;
    t->n = 0;
    t->since_loss = TIER1_TREND_SINCE_CAP;
}

/* Push one record's reduced base; write the 11-feature vector into out[11]
 * (FULL_FEATURES order). rssi must already be best-chain (sentinel on dropout). */
static void t1_trend_push(t1_trend *t, double rssi, double pkt_all,
                          double pkt_lost, double pkt_fec, double pkt_uniq,
                          double out[11]) {
    double dropout = (pkt_all == 0.0) ? 1.0 : 0.0;
    int i, k, idx;
    double rsum = 0.0, lsum = 0.0, usum = 0.0, fsum = 0.0, dsum = 0.0;
    double xbar, ybar, num, den, slope;

    if (pkt_lost > 0.0 || dropout != 0.0) {
        t->since_loss = 0.0;
    } else if (t->since_loss + 1.0 < TIER1_TREND_SINCE_CAP) {
        t->since_loss += 1.0;
    } else {
        t->since_loss = TIER1_TREND_SINCE_CAP;
    }

    t->rssi[t->head] = rssi;
    t->lost[t->head] = pkt_lost;
    t->uniq[t->head] = pkt_uniq;
    t->fec[t->head]  = pkt_fec;
    t->drop[t->head] = dropout;
    t->head = (t->head + 1) % TIER1_TREND_WINDOW;
    if (t->n < TIER1_TREND_WINDOW) t->n++;

    /* walk the ring oldest -> newest so slope's x = 0..n-1 matches Python */
    for (k = 0; k < t->n; k++) {
        idx = (t->head - t->n + k + 2 * TIER1_TREND_WINDOW) % TIER1_TREND_WINDOW;
        rsum += t->rssi[idx];
        lsum += t->lost[idx];
        usum += t->uniq[idx];
        fsum += t->fec[idx];
        dsum += t->drop[idx];
    }
    ybar = rsum / t->n;
    xbar = (t->n - 1) / 2.0;
    num = 0.0; den = 0.0;
    for (k = 0; k < t->n; k++) {
        idx = (t->head - t->n + k + 2 * TIER1_TREND_WINDOW) % TIER1_TREND_WINDOW;
        num += (k - xbar) * (t->rssi[idx] - ybar);
        den += (k - xbar) * (k - xbar);
    }
    slope = (den != 0.0) ? (num / den) : 0.0;

    /* FULL_FEATURES order: reduced base, then trend */
    out[0] = rssi;
    out[1] = pkt_all;
    out[2] = pkt_lost;
    out[3] = pkt_fec;
    out[4] = pkt_uniq;
    out[5] = slope;                                   /* rssi_slope */
    out[6] = rsum / t->n;                             /* rssi_mean */
    out[7] = (lsum + usum > 0.0) ? lsum / (lsum + usum) : 0.0;  /* loss_rate */
    out[8] = fsum / (usum + 1.0);                     /* fec_rate */
    out[9] = t->since_loss;                           /* time_since_loss */
    out[10] = dsum / t->n;                            /* dropout_frac */
    (void)i;
}

#endif /* TIER1_TREND_H */
