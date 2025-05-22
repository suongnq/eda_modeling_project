import numpy as np
import numpy as nps
import seaborn as sns
import matplotlib as mpl
import plotly.express as px
import matplotlib.pyplot as plt


def boxplot_plot(df, col, option_color, option_figsize, option_showfliers):
    mpl.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=option_figsize)
    sns.boxplot(y=df[col], color=option_color, showfliers=option_showfliers, showmeans=True)
    plt.ylabel(f'{col}')
    plt.show()

def pie_plot(df, col1, col2, title, figsize_option):
    mpl.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=figsize_option)
    plt.pie(df[col2], labels=df[col1], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Pastel1", len(df)))
    plt.title(title)
    plt.show()

def histogram_plot(df, col, color_option, figsize_option, min_xlim, max_xlim):
    plt.figure(figsize=figsize_option)
    plt.hist(df[col], bins=30, edgecolor='black', color=color_option)
    plt.title(f'Histogram of total {col} distribution per item')
    plt.xlabel(f'{col} value')
    plt.ylabel('frequency')
    plt.xlim(0, max_xlim)
    plt.grid(True)
    plt.show()

def stacked_plot(df1, df2, bar_col1, bar_col2, line_col1, line_col2, x_label, y_label, z_label, title, colors):
    ax = df1[[bar_col1, bar_col2]].plot(kind='bar', stacked=True, figsize=(8, 4), color=[colors[0], colors[1]])
    df2[line_col1].plot(ax=ax, color=colors[2], marker='o', linewidth=2, label=f'Revenue {line_col1}', secondary_y=True)
    df2[line_col2].plot(ax=ax, color=colors[3], marker='o', linewidth=2, label=f'Revenue {line_col2}', secondary_y=True)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.right_ax.set_ylabel(z_label)
    ax.set_title(title)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.legend(loc='upper left')
    ax.right_ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def map_plot(df, col_location, col_color, title):
    fig = px.choropleth(
    df,
    locations=col_location,
    locationmode='country names',
    color=col_color,
    color_continuous_scale='YlOrRd',
    title=title)

    fig.update_layout(
        width=600,
        height=400,
        legend=dict(
            x=0.8, y=0.5,
            bgcolor='rgba(255,255,255,0.6)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    fig.show()

def heat_plot(df, x_label, y_label, figsize_option):
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 10 
    plt.figure(figsize=figsize_option)
    sns.heatmap(df, annot=True, cmap="Blues", fmt=".2f", linewidths=0.2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()

def radar_comparison_plot(df1, df2, columns, labels, df1_name, df2_name, title, colors, option_figsize):
    angles = np.linspace(0, 2 * np.pi, len(columns), endpoint=False).tolist()
    angles += angles[:1]

    avg1 = df1[columns].mean().tolist()
    avg2 = df2[columns].mean().tolist()
    avg1 += avg1[:1]
    avg2 += avg2[:1]

    fig, ax = plt.subplots(figsize=option_figsize, subplot_kw=dict(polar=True))

    ax.plot(angles, avg1, color=colors[0], linewidth=2, label=df1_name)
    ax.fill(angles, avg1, color=colors[0], alpha=0.1)

    ax.plot(angles, avg2, color=colors[1], linewidth=2, label=df2_name)
    ax.fill(angles, avg2, color=colors[1], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    # ax.set_title(title, size=12, y=1.1)
    ax.legend(loc='lower right', bbox_to_anchor=(1.1, -0.3))
    plt.tight_layout()
    plt.show()

def density_plot(df1, df2, col, df1_name, df2_name, option_figsize, colors):
    plt.figure(figsize=option_figsize)
    # Overlay the density plots
    sns.kdeplot(data=df1, x=col, fill=True, common_norm=False, color= colors[0], label=df1_name)
    sns.kdeplot(data=df2, x=col, fill=True, common_norm=False, color= colors[1], label=df2_name)
    
    # Title and labels
    plt.title(f'Density of {col}')
    plt.xlabel(f'{col}')
    plt.ylabel('Density')
    
    # Add legend
    plt.legend(loc='lower right')
    
    # Show the plot
    plt.tight_layout()
    plt.show()


def combine_plot(x, mean, std, q1, median, q3 ,iqr):
    yerr = [0.2, 0.2]

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    # Vẽ boxplot, remove outliers
    medianprops = dict(linestyle='-', linewidth=2, color='yellow')
    sns.boxplot(x=x, color='lightcoral', saturation=1, medianprops=medianprops,
                flierprops={'markerfacecolor': 'mediumseagreen', 'markersize': 0},
                whis=1.5, showfliers=False,showmeans=True, ax=ax1)  # Flierprops control outliers' appearance

    ax1.set_yticks([])
    ax1.tick_params(labelbottom=True)
    ax1.set_ylim(-1, 1.5)

    # Vẽ errorbar cho Q1 và Q3
    ax1.errorbar([q1, q3], [1, 1], yerr=yerr, color='black', lw=1)

    # Thêm text annotation cho Q1, Q3, median, IQR
    ax1.text(q1, 0.6, 'Q1', ha='center', va='center', color='black')
    ax1.text(q3, 0.6, 'Q3', ha='center', va='center', color='black')
    ax1.text(median, -0.6, 'median', ha='center', va='center', color='black')
    ax1.text(median, 1.2, 'IQR', ha='center', va='center', color='black')
    ax1.text(q1 - 1.5*iqr, 0.4, 'Q1 - 1.5*IQR', ha='center', va='center', color='black')
    ax1.text(q3 + 1.5*iqr, 0.4, 'Q3 + 1.5*IQR', ha='center', va='center', color='black')

    # Vẽ KDE plot
    sns.kdeplot(x, ax=ax2)
    kdeline = ax2.lines[0]
    xs = kdeline.get_xdata()
    ys = kdeline.get_ydata()

    # Vẽ vùng cho KDE
    ylims = ax2.get_ylim()
    ax2.fill_between(xs, 0, ys, color='mediumseagreen')
    ax2.fill_between(xs, 0, ys, where=(xs >= q1 - 1.5*iqr) & (xs <= q3 + 1.5*iqr), color='skyblue')
    ax2.fill_between(xs, 0, ys, where=(xs >= q1) & (xs <= q3), color='lightcoral')

    # Đặt lại ylim cho ax2
    ax2.set_ylim(0, ylims[1])

    # Hiển thị plot
    plt.show()

