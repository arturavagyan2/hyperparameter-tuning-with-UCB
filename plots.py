import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_data():
    return pd.read_excel('titanic3.xls')

def ridgeplot(data):
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3,1)
    gs.update(hspace= -0.55)

    axes = list()
    colors = ["#022133", "#5c693b", "#51371c"]

    for idx, cls, c in zip(range(3), sorted(data['pclass'].unique()), colors):
        axes.append(fig.add_subplot(gs[idx, 0]))
        
        sns.kdeplot(x='age', data=data[data['pclass']==cls], 
                    fill=True, ax=axes[idx], cut=0, bw_method=0.25, 
                    lw=1.4, edgecolor='lightgray', hue='survived', 
                    multiple="stack", palette='PuBu', alpha=0.7
                ) 
        
        axes[idx].set_ylim(0, 0.04)
        axes[idx].set_xlim(0, 85)
        
        axes[idx].set_yticks([])
        if idx != 2 : axes[idx].set_xticks([])
        axes[idx].set_ylabel('')
        axes[idx].set_xlabel('')
        
        spines = ["top","right","left","bottom"]
        for s in spines:
            axes[idx].spines[s].set_visible(False)
            
        axes[idx].patch.set_alpha(0)
        axes[idx].text(-0.2,0,f'Class {cls}',fontweight="light", fontfamily='serif', fontsize=11,ha="right")
        if idx != 1 : axes[idx].get_legend().remove()
            
    fig.text(0.13,0.81,"Age distribution by Pclass in Titanic", fontweight="bold", fontfamily='serif', fontsize=16)

    plt.show()     

def scatterplot(data):
    survival_rate = data.groupby(['sex']).mean()[['survived']]
    male_rate = survival_rate.loc['male']
    female_rate = survival_rate.loc['female']

    male_pos = np.random.uniform(0, male_rate, len(data[(data['sex']=='male') & (data['survived']==1)]))
    male_neg = np.random.uniform(male_rate, 1, len(data[(data['sex']=='male') & (data['survived']==0)]))
    female_pos = np.random.uniform(0, female_rate, len(data[(data['sex']=='female') & (data['survived']==1)]))
    female_neg = np.random.uniform(female_rate, 1, len(data[(data['sex']=='female') & (data['survived']==0)]))

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    np.random.seed(42)

    ax.scatter(np.random.uniform(-0.3, 0.3, len(male_pos)), male_pos, color='#004c70', edgecolor='lightgray', label='Male(Survived=1)')
    ax.scatter(np.random.uniform(-0.3, 0.3, len(male_neg)), male_neg, color='#004c70', edgecolor='lightgray', alpha=0.2, label='Male(Survived=0)')

    ax.scatter(1+np.random.uniform(-0.3, 0.3, len(female_pos)), female_pos, color='#990000', edgecolor='lightgray', label='Female(Survived=1)')
    ax.scatter(1+np.random.uniform(-0.3, 0.3, len(female_neg)), female_neg, color='#990000', edgecolor='lightgray', alpha=0.2, label='Female(Survived=0)')

    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(-0.03, 1.1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Male', 'Female'], fontweight='bold', fontfamily='serif', fontsize=13)
    ax.set_yticks([], minor=False)
    ax.set_ylabel('')

    for s in ["top","right","left", 'bottom']:
        ax.spines[s].set_visible(False)

    fig.text(0.1, 1, 'Distribution of Survivors by Gender', fontweight='bold', fontfamily='serif', fontsize=15)    

    ax.legend(loc=(0.8, 0.5), edgecolor='None')
    plt.tight_layout()
    plt.show()

def corrmap(data):
    corr = data.corr()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr, 
                square=True, 
                mask=mask,
                linewidth=2.5, 
                vmax=0.4, vmin=-0.4, 
                cmap=cmap, 
                cbar=False, 
                ax=ax)

    ax.set_yticklabels(ax.get_xticklabels(), fontfamily='serif', rotation = 0, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), fontfamily='serif', rotation=90, fontsize=11)


    fig.text(0.97, 1, 'Correlation Heatmap for Titanic dataset', fontweight='bold', fontfamily='serif', fontsize=15, ha='right')    

    plt.tight_layout()
    plt.show()

def barplot(data):
    def age_band(num):
        for i in range(1, 100):
            if num < 10*i : return f'{(i-1) * 10} ~ {i*10}'

    data['age_band'] = data['age'].apply(age_band)
    titanic_age = data[['age_band', 'survived']].groupby('age_band')['survived'].value_counts().sort_index().unstack().fillna(0)
    titanic_age['Survival rate'] = titanic_age[1] / (titanic_age[0] + titanic_age[1]) * 100

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    color_map = ['#d4dddd' for _ in range(9)]
    color_map[0] = color_map[8] = '#244747'

    ax.bar(titanic_age['Survival rate'].index, titanic_age['Survival rate'], 
        color=color_map, width=0.55, 
        edgecolor='black', 
        linewidth=0.7)



    for s in ["top","right","left"]:
        ax.spines[s].set_visible(False)


    for i in titanic_age['Survival rate'].index:
        ax.annotate(f"{titanic_age['Survival rate'][i]:.02f}%", 
                    xy=(i, titanic_age['Survival rate'][i] + 2.3),
                    va = 'center', ha='center',fontweight='light', 
                    color='#4a4a4a')


    mean = data['survived'].mean() *100
    ax.axhline(mean ,color='black', linewidth=0.4, linestyle='dashdot')
    ax.annotate(f"mean : {mean :.4}%", 
                xy=('70 ~ 80', mean + 4),
                va = 'center', ha='center',
                color='#4a4a4a',
                bbox=dict(boxstyle='round', pad=0.4, facecolor='#efe8d1', linewidth=0))
        

    fig.text(0.06, 1, 'Age Band & Survival Rate', fontsize=15, fontweight='bold', fontfamily='serif')

    grid_y_ticks = np.arange(0, 101, 20)
    ax.set_yticks(grid_y_ticks)
    ax.grid(axis='y', linestyle='-', alpha=0.4)

    plt.tight_layout()
    plt.show()