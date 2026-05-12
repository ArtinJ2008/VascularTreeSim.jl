#!/usr/bin/env julia
# scale_to_rest.jl
#
# Convert a max-hyperemia (Murray-grown) coronary tree CSV to the at-rest
# (anatomical-at-100-mmHg) tree by reversing the Wong-Molloi 2008 empirical
# 1.6× resistance-vessel multiplier.
#
# Physiological model (Wong-Molloi 2008, Phys Med Biol 53:3995, eq. unnumbered
# at p. 4000):
#
#     "arterioles (diameters ⩽ 400 μm) of the LCX, LAD and RCA
#      were uniformly dilated by a factor of 1.6"
#
# That factor encodes the dilation reserve from the at-rest state (full
# autoregulatory tone in the resistance band) to maximum hyperemia
# (arteriolar smooth muscle fully relaxed, e.g. under adenosine). It was
# empirically chosen to match Pantely 1984 / Fearon 2004 in-vivo LAD
# resistance, not derived from independent physiology.
#
# The Wong-Molloi reconstruction (Kassab 1993 + Mittal 2005) goes down to
# 8 μm pre-capillary arterioles. We follow their cut-off — vessels below
# 8 μm are capillaries (no smooth muscle) and stay at their max-dilated
# diameter; vessels above 400 μm are conduit/conductance arteries
# (sparse SM + structural rigidity) and likewise don't participate in
# autoregulatory tone. Only 8 μm ≤ d ≤ 400 μm is the active resistance band.
#
# Hard band, no smoothing:
#     d_at_rest / d_max_hyper = (1 − tone)   in band
#     d_at_rest / d_max_hyper = 1.0          outside band
#
# With tone = 0.375: d_at_rest = 0.625 × d_max_hyper inside [8, 400] μm,
# equivalent to a per-vessel 1/0.625 = 1.6× dilation reserve.
#
#   D = 6 μm    -> scale = 1.000  (capillary, no SM)
#   D = 8 μm    -> scale = 0.625  (lower band edge — pre-capillary arteriole)
#   D = 100 μm  -> scale = 0.625  (in band)
#   D = 400 μm  -> scale = 0.625  (upper band edge)
#   D = 401 μm  -> scale = 1.000  (conduit unchanged)
#   D = 3.7 mm  -> scale = 1.000  (conduit unchanged)
#
# Wong-Molloi also describe a passive pressure-diameter curve (Cornelissen
# 2000, eq. 12) that adjusts each segment's diameter based on its local
# perfusion pressure. That curve is independent of and additive to this
# autoregulatory tone factor, and is not applied here — it would require
# iterative coupling with the flow solver. See hemodynamics.jl for any
# future implementation.
#
# Usage:
#   julia --project=. scripts/scale_to_rest.jl <input_csv> <output_csv> \
#         [tone=0.375] [d_low_um=8.0] [d_high_um=400.0]

using Printf

function tone_factor(d_um::Float64;
                     d_low_um::Float64=8.0,
                     d_high_um::Float64=400.0,
                     tone::Float64=0.375)
    return (d_low_um <= d_um <= d_high_um) ? (1.0 - tone) : 1.0
end

length(ARGS) >= 2 || error("Usage: scale_to_rest.jl <input_csv> <output_csv> [tone] [d_low_um] [d_high_um]")

const INPUT  = ARGS[1]
const OUTPUT = ARGS[2]
const TONE      = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.375
const D_LOW_UM  = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 8.0
const D_HIGH_UM = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 400.0

const DIAM_COL_1BASED = 14   # CSV layout: branch,segment_id,parent_segment_id,
                             # x1,y1,z1,x2,y2,z2,xmid,ymid,zmid,length_mm,
                             # diameter_um,label

# Quick preview of the tone function
println("Tone function preview:")
for d in (3700.0, 1000.0, 500.0, 401.0, 400.0, 200.0, 100.0, 50.0, 20.0, 10.0, 9.9, 6.0)
    s = tone_factor(d; d_low_um=D_LOW_UM, d_high_um=D_HIGH_UM, tone=TONE)
    @printf("  D = %7.1f μm  ->  scale = %.4f  ->  D_rest = %7.2f μm\n", d, s, d * s)
end
println()

println("Reading  : $INPUT")
println("Writing  : $OUTPUT")
println("tone     : $TONE   band: [$(D_LOW_UM), $(D_HIGH_UM)] μm   (dilation reserve = $(round(1/(1-TONE); digits=2))x)")
println("-" ^ 80)

# Diameter histogram bins (log10 μm)
const HIST_EDGES = collect(0.0:0.25:4.0)
const N_BINS = length(HIST_EDGES) - 1
counts_in  = zeros(Int, N_BINS)
counts_out = zeros(Int, N_BINS)

@inline function hist_bin!(counts::Vector{Int}, d::Float64)
    d <= 0 && return
    li = log10(d)
    idx = floor(Int, (li - HIST_EDGES[1]) / 0.25) + 1
    idx = clamp(idx, 1, N_BINS)
    counts[idx] += 1
    return
end

function process_csv(input::String, output::String;
                     d_low_um::Float64, d_high_um::Float64, tone::Float64,
                     counts_in::Vector{Int}, counts_out::Vector{Int})
    t0 = time()
    n = 0
    d_min_in, d_max_in   = Inf, -Inf
    d_min_out, d_max_out = Inf, -Inf
    io_out = open(output, "w")
    io_in = open(input, "r")
    try
        header = readline(io_in)
        println(io_out, header)
        while !eof(io_in)
            line = readline(io_in)
            isempty(line) && continue
            cols = split(line, ',', limit=15)
            d_in = parse(Float64, cols[DIAM_COL_1BASED])
            sc   = tone_factor(d_in; d_low_um=d_low_um, d_high_um=d_high_um, tone=tone)
            d_out = d_in * sc

            for i in 1:DIAM_COL_1BASED-1
                write(io_out, cols[i]); write(io_out, ',')
            end
            @printf(io_out, "%.4f", d_out)
            for i in DIAM_COL_1BASED+1:length(cols)
                write(io_out, ','); write(io_out, cols[i])
            end
            write(io_out, '\n')

            n += 1
            d_min_in  = min(d_min_in,  d_in);  d_max_in  = max(d_max_in,  d_in)
            d_min_out = min(d_min_out, d_out); d_max_out = max(d_max_out, d_out)
            hist_bin!(counts_in,  d_in)
            hist_bin!(counts_out, d_out)

            if n % 10_000_000 == 0
                dt = time() - t0
                @printf("  %.1fM rows  (%.0fk rows/s, %.0fs elapsed)\n",
                        n/1e6, n/1e3/dt, dt)
                flush(stdout)
            end
        end
    finally
        close(io_in)
        close(io_out)
    end
    dt = time() - t0
    return (n=n, dt=dt,
            d_min_in=d_min_in, d_max_in=d_max_in,
            d_min_out=d_min_out, d_max_out=d_max_out)
end

stats = process_csv(INPUT, OUTPUT;
                    d_low_um=D_LOW_UM, d_high_um=D_HIGH_UM, tone=TONE,
                    counts_in=counts_in, counts_out=counts_out)

@printf("\nWrote %d rows in %.1fs (%.0fk rows/s)\n",
        stats.n, stats.dt, stats.n/1e3/stats.dt)
@printf("Diameter range: input [%.2f, %.2f] -> output [%.2f, %.2f] μm\n",
        stats.d_min_in, stats.d_max_in, stats.d_min_out, stats.d_max_out)

println("\nDiameter histogram (log10 μm bins):")
@printf("%-15s %15s %15s %10s\n", "log10_um", "input_count", "output_count", "shift_pct")
for i in 1:N_BINS
    a = HIST_EDGES[i]; b = HIST_EDGES[i+1]
    ci, co = counts_in[i], counts_out[i]
    shift = ci > 0 ? 100.0 * (co - ci) / ci : 0.0
    @printf("[%4.2f,%4.2f)  %15d %15d   %+8.1f%%\n", a, b, ci, co, shift)
end
