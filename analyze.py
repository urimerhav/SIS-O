import glob
import pandas as pd
import numpy as np
import collections
import os
import seaborn as sns
import pylab as plt

def main():
    csvs = glob.glob('data/*.csv')
    # TODO sort csvs
    intervention_scores = collections.defaultdict(list)

    all_per_minute_sums = collections.defaultdict(list)

    for csv in csvs:
        df = pd.read_csv(csv)
        other_observed = False
        for val_row in df.values[1:32]:
            intervention_name = val_row[1]
            sum_titles_loc = np.where(df.keys() == 'Sum titles.1')[0][0]
            val_row = val_row[0:sum_titles_loc]
            minute_scoring = np.reshape(val_row[2:], [-1, 4]).astype(np.float32)[:, 0:3]
            minute_scoring[np.isnan(minute_scoring)] = 0

            per_minute_sum = np.sum(minute_scoring, axis=1).astype(np.int32)
            all_per_minute_sums[intervention_name].extend(per_minute_sum)


            nonzero_score = (per_minute_sum >= 1)

            intervention_scores[intervention_name].append(nonzero_score.sum())
            # np.histogram(per_minute_sum,[0,1,2,3,4])
            if 'Other' == intervention_name:
                other_observed = True
        print('woah')
    pass


    scores_mat = np.array(list(intervention_scores.values()))

    records = []
    for intervention_name in all_per_minute_sums:

        cntr = collections.Counter(all_per_minute_sums[intervention_name])
        ones = cntr[1]
        twos = cntr[2]
        threes = cntr[3]

        nonzeroes = ones+twos+threes

        fractions = np.round(np.array([ones, twos, threes])/nonzeroes  * 100,0).astype(np.int32)

        record = {'name': intervention_name, '1': fractions[0], '2': fractions[1], '3': fractions[2], 'cnt':nonzeroes}
        records.append(record)

    df_agreement = pd.DataFrame.from_records(records)
    df_agreement['net'] = df_agreement['3'] + df_agreement['2'] - df_agreement['1']
    df_agreement.sort_values(by='net',ascending=False)
    df_agreement.to_csv('agreement_table.csv',index=False)




    plt.imshow(scores_mat)



    all_names = list(intervention_scores.keys())

    plt.yticks(np.arange(len(all_names)), all_names)
    plt.show()


    np.array(list(intervention_scores.values()), np.int32)


if __name__ == '__main__':
    main()
