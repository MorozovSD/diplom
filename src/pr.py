import pstats

p = pstats.Stats('out.prof')

p.sort_stats('calls').print_stats()