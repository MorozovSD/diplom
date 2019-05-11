import sys


def part_progress_bar(precent, time_spend, estimation, current_max, number_count):
    text = f"\rProgress: {precent}%\t\tTime spend: {time_spend}min.\t\tEstimation: {estimation}min.\t\tCurrent max: {current_max}, for number {number_count}"
    sys.stdout.write(text)
    sys.stdout.flush()
