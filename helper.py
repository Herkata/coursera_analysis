"""
Includes the functions necessary to perform the analysis.
"""

from numpy import ndarray
import plotly.graph_objs as go
from pandas import DataFrame
import plotly.express as px
import numpy as np
from scipy.stats import norm
import pandas as pd
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.formula.api import ols


label_dict= {
    "data science": ["data", "analysis", "machine learning", "statistics", "AI ", "data science", "python", "neural networks", "deep learning", "machine learning"],
    "network": ["network", "cloud", "networking", "internet", "aws"],
    "science": ["physics", "calculus", "chemistry", "robots", "dynamics", "electronics", "big bang", "dino" "thermodynamics", "mathematical", "mathematics", "astrobiology", "bioinformatic", "solar energy"],
    "web design": ["web", "web design", "UX/UI", "user", "interface", "html5", "css3"],
    "technology": ["technology", "engineering", "IoT", "computer science", "programming", "robotics", "program", "software", 'IT ', "computing", 'technical', "programmer", "developer"],
    "business": ["business", "finanzas", "marketing", "finance", "customer", "economics","marketing", "branding", "fintech", "strategy", "strategies", "financial", "market", "accounting", "financing", "investing"],
    "education": ["education", "teaching", "learning", "pedagogy", "instruction", "teacher",],
    "health": ["health", "epidemics","medicine", "disease", "healthcare", "dentistry", "breastfeeding", "patient", "drug", "medical", "clinical", "anatomy", "care ", "dermatology", "disorder", "cancer", "infection", "gut", "gene", "bones", "immunology"],
    "psychology": ["psychology", "psychological","life ",  "mental health", "well-being", "behaviour", "happiness", "success", "cognition", "therapy", "psychiatry", "mindfulness", "parenting"],
    "humanities": ["humanities", "history", "philosophy", "literature", "art", "global", "people", "social", "human", "society", "sociological"],
    "music": ["music", "musician"],
    "energy": ["energy", "renewable"],
    "construction": ["construction"],
    "future": ["future", "futures"],
    "language": ["languge", "English", "Korean", "Spanish", "Chinese", "neurolinguistics", "Russian", "speaking", "typography", "Japanese", "tenses", "grammar", "language"],
    "security": ["security", "cybersecurity", "cyber", "secure"],
    "project management": ["project management", "agile", "scrum", "six sigma", "problem solving", "project", "projects", "lean", "product management"],
    "leadership": ["leadership", "lead", "team", "negotiation", "manager", "managers", "coach", "coaching", "motivating", "communication"],
    "legal": ["legal", "law", "criminal", "tax", "terrorism","property", "compliance", "policy", "terror" ],
    "other": []
}

def convert_str_to_num(s: str) -> int:
    """
    Convert a string with 'k' (thousand) or 'm' (million) suffix to an integer.

    Args:
        s (str): The string representation of the number.

    Returns:
        int: The converted integer value.
    """
    if "k" in s:
        num = float(s.strip("k"))
        return int(num * 1000)
    if "m" in s:
        num = float(s.strip("m"))
        return int(num * 1000000)
    return int(s)

def label_course(df: DataFrame) -> DataFrame:
    """
    This function labels each course in the DataFrame based on its title.
    The function iterates over each row in the DataFrame, converts the course title to lowercase,
    and checks if any keyword from the label_dict matches the course title.
    If a match is found, the corresponding label is assigned to the course.
    If no match is found, the course is labeled as 'other'.

    Parameters:
    df (DataFrame): The DataFrame containing the course data.
        The DataFrame should have a column named 'course_title' containing the course titles.

    Returns:
    DataFrame: The updated DataFrame with an additional column named 'label' containing the course labels.
    """
    for index, row in df.iterrows():
        course_title_lower = row['course_title'].lower()
        assigned_labels = []

    for label, keywords_list in label_dict.items():
        for keyword in keywords_list:
            if keyword.lower() in course_title_lower:
                assigned_labels.append(label)
                break

        df.at[index, 'label'] = assigned_labels[0] if assigned_labels else "other"
    return df



def label_bar_graph(series: pd.Series) -> None:
    """Creates a bar graph based on the labels and their count.

    Args:
        series (pd.Series): _description_
    """
    trace = go.Bar(
    x = series.index,
    y = series.values,
    marker=dict(colorscale='Viridis'),
    text=series.values,
    hoverinfo= 'skip')

    fig = go.Figure(data = trace)

    fig.update_layout(template = 'plotly_dark')
    fig.update_xaxes(title='Labels')
    fig.update_yaxes(title='Count')
    fig.show()


def correl_heat_map(corr_matrix: pd.DataFrame) -> None:
    """
    This function generates a correlation heatmap using Plotly for a given correlation matrix.

    Parameters:
    corr_matrix (pandas.DataFrame): A square DataFrame containing
        the correlation coefficients between numerical features.
        The index and columns of the DataFrame should represent the feature names.

    """

    vectorized_capitalize = np.vectorize(capitalize_first_word)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=vectorized_capitalize(corr_matrix.columns),
            y=vectorized_capitalize(corr_matrix.index),
            colorscale="Armyrose_r",
            zmin=-1,
            zmax=1,
            colorbar={"title": 'Correlation'},
            text=corr_matrix.values.round(2).astype(str),
            texttemplate="%{text}",
            textfont=dict(size = 16)
        )
    )

    fig.update_layout(
        title="Correlation heatmap of numerical features",
        xaxis=dict(tickfont=dict(size=16)),
        yaxis=dict(tickfont=dict(size=16), tickangle = -90),
        height=600,
        width=650,
        template="plotly_dark",
        hovermode=False,
    )

    fig.show()



def capitalize_first_word(s: str) -> str:
    """
    This function takes a string as input and capitalizes the first word of the string.

    Parameters:
    s (str): The input string that needs to have its first word capitalized.

    Returns:
    str: The input string with the first word capitalized. If the input string does not contain any words separated by underscores, the original string is returned.
    """
    words = s.split("_")
    capitalized_first_word = words[0].capitalize()
    rest_of_words = " ".join(word.lower() for word in words[1:])
    return capitalized_first_word + (" " + rest_of_words if rest_of_words else "")



def scatter_plotly(
    dataframe: DataFrame, x: ndarray, y: ndarray, outlier: DataFrame, feature: str
) -> go.Figure:
    """
    Generates a scatter plot using Plotly to visualize the distribution of a numerical feature across a dataset highlighting outliers.

    Parameters:
    dataframe (pd.DataFrame): The dataset containing the features.
    x (ndarray): The x-coordinate values.
    y (ndarray): The y-coordinate values.
    outlier (pd.DataFrame): A DataFrame containing the outlier data.
    feature (str): The name of the numerical feature to be plotted.

    Returns:
    go.Figure: A Plotly figure representing the scatter plot of the numerical feature across the dataset, with outliers highlighted.
    """
    index = dataframe.index
    data = dataframe[feature]
    mean_value = data.mean()
    std_dev = data.std()
    lower_bound = mean_value - 3 * std_dev
    upper_bound = mean_value + 3 * std_dev
    outliers = (data < lower_bound) | (data > upper_bound)

    fig = px.scatter(
        data_frame=data,
        x=x,
        y=y,
        color=y,
        color_continuous_scale="RdYlGn",
        width=1200,
        height=600,
    )

    fig.add_hline(
        y=mean_value,
        line_color="orange",
        annotation_text="Mean: {:.2f}".format(mean_value),
        annotation_position="right",
    )
    fig.add_hline(
        y=lower_bound,
        line_color="grey",
        line_dash="dash",
        annotation_text="Lower tolerance: {:.2f}".format(lower_bound),
        annotation_position="right",
    )
    fig.add_hline(
        y=upper_bound,
        line_color="grey",
        line_dash="dash",
        annotation_text="Upper tolerance: {:.2f}".format(upper_bound),
        annotation_position="right",
    )

    fig.add_trace(
        go.Scatter(
            x=index[outliers],
            y=data[outliers],
            mode="markers",
            marker=dict(color="red", symbol="x", size=10),
            name="Outlier",
            hoverinfo="skip",
        )
    )

    y_min, y_max = data.min(), data.max()
    y_values = np.linspace(y_min, y_max, 100)
    normal_dist = norm.pdf(y_values, mean_value, std_dev)

    normal_dist = normal_dist / normal_dist.max() * (max(x) / 10)

    fig.add_trace(
        go.Scatter(
            x=-normal_dist,
            y=y_values,
            mode="lines",
            line=dict(color="blue"),
            name="Normal Distribution",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        yaxis_title=capitalize_first_word(feature),
        xaxis_title="Course index",
        showlegend=False,
        margin=dict(r=200, l=100),
        coloraxis_showscale=False,
        template="plotly_dark",
    )

    fig.update_yaxes(
        ticklabelposition="outside left",
    )
    fig.update_xaxes(range=[-100, 900])

    return fig

def bar_plot(df: DataFrame, category_cols: pd.Index) -> None:
    """
    Generates a series of bar plots to visualize the distribution of categorical features in the DataFrame.

    Parameters:
    df (DataFrame): A pandas DataFrame containing the dataset with columns representing categorical features.
    category_cols (Index): An Index[str] variable containing the categorical features.

    """

    fig = make_subplots(
        rows=1,
        cols=len(category_cols),
        subplot_titles=[capitalize_first_word(title) for title in category_cols]
    )
    vectorized_capitalize = np.vectorize(capitalize_first_word)

    for i, feat in enumerate(category_cols):
        cat_count = df[feat].value_counts()
        fig.add_trace(
            go.Bar(
                x=vectorized_capitalize(cat_count.index),
                y=cat_count.values,
                text=cat_count.values,
                name=capitalize_first_word(feat)
            ),
            row=1,
            col=i + 1
        )

        fig.update_layout(
            title="Bar plots of categorical features",
            showlegend=False,
            height=400,
            width=600 * len(category_cols),
            template="plotly_dark",
            hovermode=False,
            font = dict(size = 16)
        )

    fig.show()

def plot_lan(data: DataFrame) -> None:
    """Generates a bar plot to display the number of Non-English courses.

    Args:
        data (DataFrame): The original dataframe.
    """
    fig = px.bar(data_frame=data.language.value_counts()[data.language.value_counts()>= 2], y = 'count', text = 'count')

    fig.update_layout(template = 'plotly_dark',
                      hovermode=False,
                      title = 'Non-English Courses'
                      )
    fig.update_xaxes(title = 'Non-English languages')
    fig.update_yaxes(title = 'Count')
    fig.show()

def compare_lang(df: DataFrame) -> None:
    """Generates a bar graph comparing the ratio of the language being spoken worldwide and it's share in the course dataset.

    Args:
        df (DataFrame): the DataFrame containing the language, and percentages for worldwide and course data.
    """
    fig = go.Figure(data=[
    go.Bar(name='Worldwide %', x=df.language, y=df['worldwide'], text = df['worldwide']),
    go.Bar(name='In dataset %', x=df.language, y=df['in dataset'], text= df['in dataset'])
])
    fig.update_layout(
        barmode = 'group',
        template = 'plotly_dark',
        hovermode = False,
        title = 'Comparison of languages spoken worldwide and course languages')
    fig.update_yaxes(ticksuffix =  " %")
    fig.show()

def course_count_rating_plot(df: pd.DataFrame, x: int) -> None:
    """
    This function generates a bar plot to visualize the relationship between the count of courses per organization and their average rating.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the dataset with columns 'course_organization' and 'course_rating'.
    x (int): A threshold value for the minimum number of courses per organization to be considered for the plot.

    """
    course_per_inst = df["course_organization"].value_counts()
    course_per_inst = course_per_inst[course_per_inst > x]

    filtered_data = df[df["course_organization"].isin(course_per_inst.index)]
    filtered_data = filtered_data.groupby("course_organization")["course_rating"].agg(
        "mean"
    )

    fig = px.bar(
        data_frame=course_per_inst,
        color=filtered_data,
        text_auto=True,
        labels=dict(color="Rating"),
    )

    fig.update_layout(
        template="plotly_dark",
        hovermode=False,
        xaxis_title="Course organization",
        yaxis_title="Course count",
        font = dict(size = 16)
    )

    fig.update_xaxes(tickangle = 20)

    fig.update_traces(width=0.45)

    fig.show()



def perform_anova_and_visualize(
    data: pd.DataFrame, categorical_feature: str, numerical_feature: str
) -> None:
    """
    Performs a one-way ANOVA on the given dataset and visualizes the results.

    Parameters:
    data (pd.DataFrame): The dataset containing the features.
    categorical_feature (str): The name of the categorical feature.
    numerical_feature (str): The name of the numerical feature.

    """

    course_per_inst = data["course_organization"].value_counts()
    course_per_inst = course_per_inst[course_per_inst > 20]

    filtered_data = data[data["course_organization"].isin(course_per_inst.index)]

    # Perform ANOVA
    formula = f"{numerical_feature} ~ C({categorical_feature})"
    model = ols(formula, data=filtered_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Print ANOVA results
    #print(anova_table)

    # Create a box plot using Plotly
    fig = px.box(
        filtered_data,
        x=categorical_feature,
        y=numerical_feature,
        #points="all",
        title=f"{capitalize_first_word(numerical_feature)} distribution across {capitalize_first_word(categorical_feature)} categories",
        labels={
            categorical_feature: categorical_feature.replace("_", " ").title(),
            numerical_feature: numerical_feature.replace("_", " ").title(),
        },
    )

    # Customize the layout for scrolling
    fig.update_layout(
        boxmode="group",
        xaxis=dict(
            tickangle=30,
            automargin=True,
            showgrid=False,
            zeroline=False,
            type="category",
            categoryorder="total descending",
        ),
        height=600,
        margin=dict(l=100, r=100, t=100, b=100),
    )

    # Add ANOVA results as annotations
    f_statistic = anova_table["F"].iloc[0]
    p_value = anova_table["PR(>F)"].iloc[0]
    fig.add_annotation(
        x=0,  # X position for annotation (left side)
        y=1.1,
        text=f"F-statistic: {f_statistic:.2f}, p-value: {p_value:.3e}",
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="white"),
        bordercolor="white",
        borderwidth=1,
    )

    fig.update_layout(template="plotly_dark",
                      font = dict(size=14))

    fig.show()
