#!/usr/bin/env julia
# scale_to_rest.jl
#
# Convert a max-dilated coronary tree CSV to an at-rest tree by applying a
# diameter-dependent arteriolar tone factor.
#
# Why: Murray's law (d_parent³ = sum d_child³) gives the energetically optimal
# vessel diameters at the point of maximum dilation (smooth muscle fully
# relaxed). In real physiology, the at-rest state is reached by constricting
# small arteries / arterioles via myogenic + metabolic + neural tone. The
# largest dilation reserve sits at ~100 μm arterioles (CFR primary regulators);
# capillaries (no smooth muscle) and large conduit arteries (sparse smooth
# muscle, structural rigidity) barely change.
#
# Model: bell curve in log10(D) centered at 100 μm, peak constriction
# fraction `tone_max`. Default tone_max=0.4 → at peak D_at_rest = 0.6·D_max.
# Width=0.4 (log10 units) tapers smoothly across the arteriolar band.
#
#   D = 6 μm    -> scale ≈ 0.996  (capillary unchanged)
#   D = 50 μm   -> scale ≈ 0.70
#   D = 100 μm  -> scale ≈ 0.60   (peak arteriole constriction)
#   D = 500 μm  -> scale ≈ 0.91
#   D = 1 mm    -> scale ≈ 0.98
#   D = 3.7 mm  -> scale ≈ 1.000  (conduit unchanged)
#
# Usage:
#   julia --project=. scripts/scale_to_rest.jl <input_csv> <output_csv> \
#         [tone_max=0.4] [width=0.4] [center_um=100]

using Printf

function tone_factor(d_um::Float64;
                     center_log10::Float64=2.0,
                     width::Float64=0.4,
                     tone_max::Float64=0.4)
    z = (log10(d_um) - center_log10) / width
    g = exp(-0.5 * z * z)
    return 1.0 - tone_max * g
end

length(ARGS) >= 2 || error("Usage: scale_to_rest.jl <input_csv> <output_csv> [tone_max] [width] [center_um]")

const INPUT  = ARGS[1]
const OUTPUT = ARGS[2]
const TONE_MAX     = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.4
const WIDTH        = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.4
const CENTER_LOG10 = length(ARGS) >= 5 ? log10(parse(Float64, ARGS[5])) : 2.0

const DIAM_COL_1BASED = 14   # CSV layout: branch,segment_id,parent_segment_id,
                             # x1,y1,z1,x2,y2,z2,xmid,ymid,zmid,length_mm,
                             # diameter_um,label

# Quick preview of the tone function
println("Tone function preview:")
for d in (3700.0, 1000.0, 500.0, 200.0, 100.0, 50.0, 20.0, 10.0, 6.0)
    s = tone_factor(d; center_log10=CENTER_LOG10, width=WIDTH, tone_max=TONE_MAX)
    @printf("  D = %7.1f μm  ->  scale = %.4f  ->  D_rest = %7.2f μm\n", d, s, d * s)
end
println()

println("Reading  : $INPUT")
println("Writing  : $OUTPUT")
println("tone_max : $TONE_MAX   width: $WIDTH (log10)   center: $(10^CENTER_LOG10) μm")
println("-" ^ 80)

# Diameter histogram bins (log10 μm)
const HIST_EDGES = collect(0.0:0.25:4.0)
const N_BINS = length(HIST_EDGES) - 1
counts_in  = zeros(Int, N_BINS)
counts_out = zeros(Int, N_BINS)
d_min_in,  d_max_in  = Inf, -Inf
d_min_out, d_max_out = Inf, -Inf

@inline function hist_bin!(counts::Vector{Int}, d::Float64)
    d <= 0 && return
    li = log10(d)
    idx = floor(Int, (li - HIST_EDGES[1]) / 0.25) + 1
    idx = clamp(idx, 1, N_BINS)
    counts[idx] += 1
    return
end

function process_csv(input::String, output::String;
                     center_log10::Float64, width::Float64, tone_max::Float64,
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
            sc   = tone_factor(d_in; center_log10=center_log10, width=width, tone_max=tone_max)
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
                    center_log10=CENTER_LOG10, width=WIDTH, tone_max=TONE_MAX,
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
