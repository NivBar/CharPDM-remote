import pandas as pd
import config
import utils

if __name__ == '__main__':
    if config.data_exist:
        df = pd.read_csv("comp_dataset.csv")
    else:
        df = utils.data_set_creation()
        for q in set(df.query_id):
            q_df = df[df.query_id == q]
            for a in set(q_df.author_id):
                a_df = q_df[q_df.author_id == a]
                min_, max_ = min(a_df.round_number.astype(int)) + 1, max(a_df.round_number.astype(int)) + 1
                for i in range(min_, max_):
                    idx = a_df[a_df.round_number == str(i)].index[0]
                    df.at[idx, "DELTA"] = int(a_df[a_df.round_number == str(i - 1)]["POS"]) - int(
                        a_df[a_df.round_number == str(i)]["POS"])
                for i in range(min_ + 1, max_):
                    idx = a_df[a_df.round_number == str(i)].index[0]
                    df.at[idx, "DELTA_2"] = int(a_df[a_df.round_number == str(i - 2)]["POS"]) - int(
                        a_df[a_df.round_number == str(i)]["POS"])
                for i in range(min_ + 2, max_):
                    idx = a_df[a_df.round_number == str(i)].index[0]
                    df.at[idx, "DELTA_3"] = int(a_df[a_df.round_number == str(i - 3)]["POS"]) - int(
                        a_df[a_df.round_number == str(i)]["POS"])
        df.fillna(0, inplace=True)
        df = df.astype({'DELTA': 'int', 'DELTA_2': 'int', 'DELTA_3': 'int'})
        df.to_csv("comp_dataset.csv", index=False)
    # improvements df creation
    if config.data_exist:
        improvements = pd.read_csv("improvements_data.csv")
    else:
        improvements = df[((df["DELTA"] > 0) | (df["DELTA_2"] > 0) | (df["DELTA_3"] > 0)) & (df["KSREL"] != 3)].sort_values(
            by="DELTA", ascending=False)
        rows = []
        for idx, row in improvements.iterrows():
            for suff in ["", "_2", "_3"]:
                if row["DELTA" + suff] > 0:
                    new_row = dict(row)
                    new_row["improved_from"] = df[
                        (df["query_id"] == row["query_id"]) & (df["author_id"] == row["author_id"]) & (
                                df["round_number"] == row["round_number"] - 1)]["TEXT"].values[0]
                    rows.append(new_row)
        improvements = pd.DataFrame(rows)
        improvements["dup_val"] = improvements["TEXT"] + improvements["improved_from"]
        improvements = improvements.drop_duplicates(subset=['dup_val']).drop(columns=["dup_val"])
        improvements.to_csv("improvements_data.csv", index=False)

    # tops df creation
    if config.data_exist:
        tops = pd.read_csv("tops_data.csv")
    else:
        tops = df[(df["POS"].astype(str).str.contains("1|2")) & (df["KSREL"] != 3)].sort_values(["query_id", "round_number"],
                                                                                                ascending=False)
        tops = tops[tops.duplicated(["round_number", "query_id"], keep=False)]
        rows = []
        for idx, row in tops[tops.POS == 1].iterrows():
            new_row = dict(row)
            texts = \
                tops[(tops["query_id"] == row["query_id"]) & (tops["round_number"] == row["round_number"]) & (
                            row["POS"] < tops["POS"])][
                    "TEXT"].values
            new_row["improved_from"] = texts[0]
            rows.append(new_row)
        tops = pd.DataFrame(rows)
        tops["dup_val"] = tops["TEXT"] + tops["improved_from"]
        tops = tops.drop_duplicates(subset=['dup_val']).drop(columns=["dup_val"])
        tops.to_csv("tops_data.csv", index=False)
