/*
 * _rainflow.c — High-performance 4-point rainflow cycle counting.
 *
 * Compiled as a shared library and loaded via ctypes from Python.
 * The algorithm is O(n) amortised: each turning point is pushed onto
 * the stack at most once and popped at most once.
 *
 * Build:
 *   gcc -O3 -march=native -shared -fPIC -o _rainflow.so _rainflow.c
 */

#include <math.h>
#include <stdlib.h>

/*
 * rainflow_4point
 *
 * Inputs:
 *   tp       — array of turning-point values (double, length n)
 *   n        — number of turning points
 *
 * Pre-allocated outputs (caller must allocate at least n elements each):
 *   full_r   — full-cycle ranges
 *   full_mx  — full-cycle max values
 *   full_mn  — full-cycle min values
 *   half_r   — half-cycle ranges
 *   half_mx  — half-cycle max values
 *   half_mn  — half-cycle min values
 *
 * Output counts (written by this function):
 *   out_fc   — number of full cycles written
 *   out_hc   — number of half cycles written
 */
void rainflow_4point(
    const double *tp, int n,
    double *full_r,  double *full_mx,  double *full_mn,
    double *half_r,  double *half_mx,  double *half_mn,
    int *out_fc, int *out_hc)
{
    /* Stack of indices into tp */
    int *stack = (int *)malloc((size_t)n * sizeof(int));
    if (!stack) { *out_fc = 0; *out_hc = 0; return; }

    int sp = 0;   /* stack pointer (number of elements on stack) */
    int fc = 0;   /* full cycle count */
    int i, j;
    double v1, v2, v3, v4, inner, outer, a, b;

    for (i = 0; i < n; i++) {
        stack[sp++] = i;

        while (sp >= 4) {
            v1 = tp[stack[sp - 1]];
            v2 = tp[stack[sp - 2]];
            v3 = tp[stack[sp - 3]];
            v4 = tp[stack[sp - 4]];

            inner = fabs(v2 - v3);
            outer = fabs(v1 - v4);

            if (inner <= outer) {
                full_r[fc] = inner;
                if (v2 >= v3) {
                    full_mx[fc] = v2;
                    full_mn[fc] = v3;
                } else {
                    full_mx[fc] = v3;
                    full_mn[fc] = v2;
                }
                fc++;
                /* Remove inner two points: keep outer two */
                stack[sp - 3] = stack[sp - 1];
                sp -= 2;
            } else {
                break;
            }
        }
    }

    /* Remaining stack entries form half cycles */
    int hc = 0;
    for (j = 0; j < sp - 1; j++) {
        a = tp[stack[j]];
        b = tp[stack[j + 1]];
        half_r[hc] = fabs(b - a);
        if (a >= b) {
            half_mx[hc] = a;
            half_mn[hc] = b;
        } else {
            half_mx[hc] = b;
            half_mn[hc] = a;
        }
        hc++;
    }

    *out_fc = fc;
    *out_hc = hc;

    free(stack);
}
