import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_mean_val_comparisons(dict1, dict2, name1, name2):
    '''
    plots a bar graph that compares the mean absolute error of two segmentation sources relative to a gold standard source
    
    inputs:
    dict1: dictionary specifying the mean absolute disagreement between segmentation source 1 and gold standard
    dict1: dictionary specifying the mean absolute disagreement between segmentation source 2 and gold standard
    name1: name of segmentation source 1
    name2: name of segmentation source 2
    
    '''
    labels = np.sort(list(dict1.keys()))
    values1 = [dict1[i] for i in labels]
    values2 = [dict2[i] for i in labels]
    
    means1 = [np.round(i[0],2) for i in values1]
    error1 = [np.round(i[1],2) for i in values1]
    means2 = [np.round(i[0],2) for i in values2]
    error2 = [np.round(i[1],2) for i in values2]

    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, means1, width, yerr=error1, label=name1)
    rects2 = ax.bar(x + width/2, means2, width, yerr=error2, label=name2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean error relative to gold standard +/- StD (ms)', size = 20)
    ax.set_title('Mean absolute error in average T2 value for each region', size = 30)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, size = 20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15) 

    ax.legend(prop={'size': 20})


#     def autolabel(rects):
#         """Attach a text label above each bar in *rects*, displaying its height."""
#         for rect in rects:
#             height = rect.get_height()
#             ax.annotate('{}'.format(height),
#                         xy=(rect.get_x() + rect.get_width() / 2, height),
#                         xytext=(0, 3),  # 3 points vertical offset
#                         textcoords="offset points",
#                         ha='center', va='bottom', size = 15)


#     autolabel(rects1)
#     autolabel(rects2)
    fig.set_size_inches(30, 9, forward=True)

    fig.tight_layout()

    plt.show()