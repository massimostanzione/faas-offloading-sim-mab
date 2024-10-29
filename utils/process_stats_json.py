import json
import sys


def sum_dict (d):
    s=0
    for k in d:
        s+=d[k]
    return s


def process_json (stats):
    _cum_cold_starts = 0

    for s in stats:
        t = s["_Time"]

        cum_cold_starts = sum_dict(s["cold_starts"])
        cold_starts = cum_cold_starts-_cum_cold_starts
        _cum_cold_starts = cum_cold_starts

        utility = s["utility"]
        cost = s["cost"]


        print(f"{t},{cold_starts},{utility}")



def main():
    json_files = sys.argv[1:] if len(sys.argv) > 1 else ["./stats.txt"]

    for json_file in json_files:
        #
        # process single json file
        #
        with open(json_file, "r") as jsonf:
            stats = json.load(jsonf)
            stats = sorted(stats, key=lambda x: x["_Time"])
            print(f"#{json_file}")
            process_json (stats)
            print("\n\n")

if __name__ == "__main__":
    main()
