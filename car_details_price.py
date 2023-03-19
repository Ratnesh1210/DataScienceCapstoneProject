{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Ratnesh1210/Car-details-price-prediction/blob/main/car_details_price.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wh2hreVP3_Nn"
      },
      "source": [
        "# **Project Name**    -  Car Price Prediction Using Regression model"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lScro1gQ4PPn"
      },
      "source": [
        "##### **Project Type**    - Regression\n",
        "##### **Contribution**    - Individual\n",
        "##### **Cntributor -** - Ratnesh Verma\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9_ThoPfp44VF"
      },
      "source": [
        "# **Project Summary -**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4nfXq3cL2DTc"
      },
      "source": [
        "This dataset comprises a wealth of information regarding used cars, including details such as the year in which the car was manufactured, the selling price, the amount of kilometers driven, the type of fuel used, the type of seller (whether a dealer or an individual), the type of transmission (automatic or manual), and the number of previous owners. \n",
        "\n",
        "the task at hand is to apply various machine learning regression models to the provided used car dataset, with the objective of accurately predicting the selling price of the cars based on the available features. This could involve using multiple linear regression, polynomial regression, decision tree regression, random forest regression, or other suitable regression models. The ultimate goal is to identify the model or combination of models that yields the most accurate and reliable price predictions for the used cars in the dataset.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4mbK17dx5Bt2"
      },
      "source": [
        "# **GitHub Link -**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Js8Uv_9A5Fzy"
      },
      "source": [
        "**Problem Statement**\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JoOiKafh6OuD"
      },
      "source": [
        "the task at hand is to apply various machine learning regression models to the provided used car dataset, with the objective of accurately predicting the selling price of the cars based on the available features. This could involve using multiple linear regression, polynomial regression, decision tree regression, random forest regression, or other suitable regression models. The ultimate goal is to identify the model or combination of models that yields the most accurate and reliable price predictions for the used cars in the dataset."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oRk84paX5Lho"
      },
      "source": [
        "# ***Let's Begin !***"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CO4DR47-61Ne"
      },
      "source": [
        "## ***1. Know Your Data***"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6FJjusfBnOuW"
      },
      "outputs": [],
      "source": [
        "# Import Libraries\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import scipy\n",
        "import matplotlib as mpl\n",
        "import matplotlib.pyplot as plt\n",
        "%matplotlib inline\n",
        "import seaborn as sns\n",
        "from datetime import datetime\n",
        "\n",
        "import warnings    \n",
        "warnings.filterwarnings('ignore')\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QnaqS7hFmH9y",
        "outputId": "a5ce6236-c15a-437d-c53c-96307073ae8c"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "#mounting drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hIH92z6TtlL-"
      },
      "source": [
        "### Dataset First View"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "p93CbieCkVqm"
      },
      "outputs": [],
      "source": [
        "#Importing data set\n",
        "df =pd.read_csv(\"/content/drive/MyDrive/CAR DETAILS (1).csv\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0TFVpATvnK3Z",
        "outputId": "5bb7396c-1637-4c9a-b698-c5d8039271f0"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-1ac509b2-733f-4357-a770-7bb8cf6bc23d\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>name</th>\n",
              "      <th>year</th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Maruti 800 AC</td>\n",
              "      <td>2007</td>\n",
              "      <td>60000</td>\n",
              "      <td>70000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Maruti Wagon R LXI Minor</td>\n",
              "      <td>2007</td>\n",
              "      <td>135000</td>\n",
              "      <td>50000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Hyundai Verna 1.6 SX</td>\n",
              "      <td>2012</td>\n",
              "      <td>600000</td>\n",
              "      <td>100000</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Datsun RediGO T Option</td>\n",
              "      <td>2017</td>\n",
              "      <td>250000</td>\n",
              "      <td>46000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Honda Amaze VX i-DTEC</td>\n",
              "      <td>2014</td>\n",
              "      <td>450000</td>\n",
              "      <td>141000</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>Second Owner</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-1ac509b2-733f-4357-a770-7bb8cf6bc23d')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-1ac509b2-733f-4357-a770-7bb8cf6bc23d button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-1ac509b2-733f-4357-a770-7bb8cf6bc23d');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "                       name  year  selling_price  km_driven    fuel  \\\n",
              "0             Maruti 800 AC  2007          60000      70000  Petrol   \n",
              "1  Maruti Wagon R LXI Minor  2007         135000      50000  Petrol   \n",
              "2      Hyundai Verna 1.6 SX  2012         600000     100000  Diesel   \n",
              "3    Datsun RediGO T Option  2017         250000      46000  Petrol   \n",
              "4     Honda Amaze VX i-DTEC  2014         450000     141000  Diesel   \n",
              "\n",
              "  seller_type transmission         owner  \n",
              "0  Individual       Manual   First Owner  \n",
              "1  Individual       Manual   First Owner  \n",
              "2  Individual       Manual   First Owner  \n",
              "3  Individual       Manual   First Owner  \n",
              "4  Individual       Manual  Second Owner  "
            ]
          },
          "execution_count": 4,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df.head()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JopB3U81tpgN"
      },
      "source": [
        "### Dataset Rows & Columns count"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LzGOshuOtvle",
        "outputId": "318b6e66-ab4a-4ed7-fc45-45ed46b6767c"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "The number of rows in data is: 4340\n",
            "The number of columns in data is 8\n"
          ]
        }
      ],
      "source": [
        "# Dataset Rows & Columns count\n",
        "print('The number of rows in data is:',df.shape[0])\n",
        "print('The number of columns in data is',len(list(df.columns)))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aQkqWisBt0O-"
      },
      "source": [
        "### Dataset Information"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MWyd58bAt141",
        "outputId": "4f65aff2-678b-4817-a755-763a418953c0"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 4340 entries, 0 to 4339\n",
            "Data columns (total 8 columns):\n",
            " #   Column         Non-Null Count  Dtype \n",
            "---  ------         --------------  ----- \n",
            " 0   name           4340 non-null   object\n",
            " 1   year           4340 non-null   int64 \n",
            " 2   selling_price  4340 non-null   int64 \n",
            " 3   km_driven      4340 non-null   int64 \n",
            " 4   fuel           4340 non-null   object\n",
            " 5   seller_type    4340 non-null   object\n",
            " 6   transmission   4340 non-null   object\n",
            " 7   owner          4340 non-null   object\n",
            "dtypes: int64(3), object(5)\n",
            "memory usage: 271.4+ KB\n"
          ]
        }
      ],
      "source": [
        "# Dataset Info\n",
        "df.info()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7irSj-mOuCYY"
      },
      "source": [
        "#### Duplicate Values"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lnsxP2RruDcn",
        "outputId": "ffb637fc-200e-4aa1-b8cd-502dbddf6718"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "The number of duplicates rows in the data is: 763\n"
          ]
        }
      ],
      "source": [
        "# Dataset Duplicate Value Count\n",
        "# df.duplicated().sum()\n",
        "duplicate_rows_in_store_data = df.duplicated().sum()\n",
        "print('The number of duplicates rows in the data is:',duplicate_rows_in_store_data)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "u1EMK8ZUuRrV"
      },
      "source": [
        "#### Missing Values/Null Values"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2NP367zOt-6e",
        "outputId": "77c07742-0c78-4c62-d4ba-082d945c6fb9"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "name             0\n",
              "year             0\n",
              "selling_price    0\n",
              "km_driven        0\n",
              "fuel             0\n",
              "seller_type      0\n",
              "transmission     0\n",
              "owner            0\n",
              "dtype: int64"
            ]
          },
          "execution_count": 8,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Missing Values/Null Values Count\n",
        "df.isnull().sum()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yjQFUDnduok-"
      },
      "source": [
        "### What did you know about your dataset?"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "A69tlmKHuusu"
      },
      "source": [
        "1. -There is no null value present in the data set.\n",
        "2. -There are 8 columns in the dataset.\n",
        "3. -The number of duplicate row in the dataset is 763. "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-lQDI0QkuvoV"
      },
      "source": [
        "## ***2. Understanding Your Variables***"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bpaGd0EouqGO",
        "outputId": "628da212-329b-4a41-aab7-d39919f00fa1"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']\n"
          ]
        }
      ],
      "source": [
        "# Dataset Columns\n",
        "print(list(df.columns))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fErLQCddLTQx",
        "outputId": "e0873f50-29af-4035-f3be-fc8586e5bafd"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 4340 entries, 0 to 4339\n",
            "Data columns (total 8 columns):\n",
            " #   Column         Non-Null Count  Dtype \n",
            "---  ------         --------------  ----- \n",
            " 0   name           4340 non-null   object\n",
            " 1   year           4340 non-null   int64 \n",
            " 2   selling_price  4340 non-null   int64 \n",
            " 3   km_driven      4340 non-null   int64 \n",
            " 4   fuel           4340 non-null   object\n",
            " 5   seller_type    4340 non-null   object\n",
            " 6   transmission   4340 non-null   object\n",
            " 7   owner          4340 non-null   object\n",
            "dtypes: int64(3), object(5)\n",
            "memory usage: 271.4+ KB\n"
          ]
        }
      ],
      "source": [
        "# checking information about the data type of the variable\n",
        "df.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZNGrZWRwu1uN",
        "outputId": "88ee5273-3a5d-4b7a-9a97-ed1262f7dec1"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-02794f71-8ae6-4021-a0d4-b1d5bab2eb0f\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>year</th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>4340.000000</td>\n",
              "      <td>4.340000e+03</td>\n",
              "      <td>4340.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>2013.090783</td>\n",
              "      <td>5.041273e+05</td>\n",
              "      <td>66215.777419</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>4.215344</td>\n",
              "      <td>5.785487e+05</td>\n",
              "      <td>46644.102194</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>1992.000000</td>\n",
              "      <td>2.000000e+04</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>2011.000000</td>\n",
              "      <td>2.087498e+05</td>\n",
              "      <td>35000.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>2014.000000</td>\n",
              "      <td>3.500000e+05</td>\n",
              "      <td>60000.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>2016.000000</td>\n",
              "      <td>6.000000e+05</td>\n",
              "      <td>90000.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>2020.000000</td>\n",
              "      <td>8.900000e+06</td>\n",
              "      <td>806599.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-02794f71-8ae6-4021-a0d4-b1d5bab2eb0f')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-02794f71-8ae6-4021-a0d4-b1d5bab2eb0f button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-02794f71-8ae6-4021-a0d4-b1d5bab2eb0f');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "              year  selling_price      km_driven\n",
              "count  4340.000000   4.340000e+03    4340.000000\n",
              "mean   2013.090783   5.041273e+05   66215.777419\n",
              "std       4.215344   5.785487e+05   46644.102194\n",
              "min    1992.000000   2.000000e+04       1.000000\n",
              "25%    2011.000000   2.087498e+05   35000.000000\n",
              "50%    2014.000000   3.500000e+05   60000.000000\n",
              "75%    2016.000000   6.000000e+05   90000.000000\n",
              "max    2020.000000   8.900000e+06  806599.000000"
            ]
          },
          "execution_count": 11,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Dataset Describe\n",
        "df.describe()\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YY6Pa_SxvKAG"
      },
      "source": [
        "### Check Unique Values for each variable."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YwMZmCDIUWk8",
        "outputId": "8d10e14f-8c02-4e90-a5a7-27ff21179882"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "1491"
            ]
          },
          "execution_count": 12,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "#checking number of unique cars available in the data set\n",
        "df['name'].nunique()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "O2sWw3LuvCgn",
        "outputId": "1f4cb74c-6f12-4f2a-a000-567ac7c242e2"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "The unique values in name are ['Maruti 800 AC' 'Maruti Wagon R LXI Minor' 'Hyundai Verna 1.6 SX' ...\n",
            " 'Mahindra Verito 1.5 D6 BSIII'\n",
            " 'Toyota Innova 2.5 VX (Diesel) 8 Seater BS IV'\n",
            " 'Hyundai i20 Magna 1.4 CRDi']\n",
            "The unique values in year are [2007 2012 2017 2014 2016 2015 2018 2019 2013 2011 2010 2009 2006 1996\n",
            " 2005 2008 2004 1998 2003 2002 2020 2000 1999 2001 1995 1997 1992]\n",
            "The unique values in selling_price are [  60000  135000  600000  250000  450000  140000  550000  240000  850000\n",
            "  365000  260000 1650000  585000 1195000  390000 1964999 1425000  975000\n",
            " 1190000  930000  525000 1735000 1375000  900000 1300000 1400000  229999\n",
            " 1550000 1250000  625000 1050000  560000  290000  275000  411000  150000\n",
            "  500000  100000  725000  401000  750000  310000  665000  465000  160000\n",
            "  675000  300000   70000  151000  280000  350000  570000  125000  130000\n",
            "  925000  200000  248000   80000  650000  495000  371000 1025000 8150000\n",
            "  325000 1470000 2800000  210000 1150000 4500000 2750000 1975000  175000\n",
            " 2500000  628000  399000  315000  780000  434000  690000  555000  120000\n",
            "  165000   95000  800000  840000  490000  400000 1000000  530000   40000\n",
            "   75000  540000  700000  430000   65000  195000  170000  225000   99000\n",
            "  620000 2550000  320000  810000  282000   72000  640000  380000 1500000\n",
            "  434999  190000 2900000  425000  265000  890000  685000  940000  590000\n",
            "  385000 2000000  235000   52000   89999  180000  285000 1075000   90000\n",
            "  220000  110000  880000  115999  360000  680000  860000  270000  395000\n",
            "  624000  345000  106000 1800000  575000  370000   50000   55000  755000\n",
            "  720000 1100000  159000  335000  185000  470000  145000  595000 1600000\n",
            "  105000  409999  215000  475000  330000 1044999 1350000  420000  760000\n",
            "   43000 1850000 1125000  133000  352000  520000  509999  556000  484999\n",
            "  565000  295000 2050000 1475000 4400000  670000  770000  775000 1725000\n",
            " 2150000 3800000 1580000 4950000  535000  239000 2600000  114999  200999\n",
            "  710000  969999  155000  138000  311000   58000  183000  825000  299000\n",
            "  639000  415000 1199000  699000  269000  249000 1549000  254999  211000\n",
            "  599000 4000000 1200000   98000  790000 1700000   68000  875000 1330000\n",
            "  919999  611000  711000  851000  610000  744000  480000  950000   85000\n",
            "  615000  227000  222000  735000  271000 1490000  455000  421000 2700000\n",
            " 4700000 1900000 1770000  660000  716000  147000 1140000 3050000  375000\n",
            " 1950000  340000 3100000  245000  715000 1750000 3500000  835000 2490000\n",
            " 1015000   91200 2400000  635000  302000  204999  341000  819999  351000\n",
            "  630000 1085000  580000   78000 3200000  695000  355000  619000   81000\n",
            "  486000  802000 2300000  287000  250999   45000 1485000 1825000 3256000\n",
            "  451000  149000  163000  419000  990000  346000  509000   69000 1380000\n",
            "  256000   97000  199000 2595000  730000  368000  545000  641000  784000\n",
            "  324000 2100000  305000  221000  828999 1119000  746000 1030000 1334000\n",
            "  811999 1331000  852000  830000  213000   35000  869999  178000  515000\n",
            "  312000  111000  774000  148000   57000  284000  349000  458000  381000\n",
            "  751000  782000  321000   92800  291000   73000  655000  263000  217000\n",
            "  539000  142000  910000  740000  164000  999000   56000 3899000  440000\n",
            "  238000 1295000  541000  894999  844999  288000 1225000 1010000   30000\n",
            "  396000  281000   93000  459999   88000   22000   79000  198000  182000\n",
            "  861999  836000  696000  596000  612000   20000   61000  511000 1230000\n",
            "  426000   62000 1450000   71000 2200000 1249000 1240000 1068000 1189000\n",
            "  363000  821000  815000  738000  765000  516000  134000  347000 2650000\n",
            " 2675000  359000  980000  707000  471000  377000  763000  701000  277000\n",
            "  936999   82000  799000 1451000 1575000   78692  479000   48000  121000\n",
            "  785000  173000 4800000  587000  123000 1290000  193000  721000 1040000\n",
            " 2349000 1165000   42000 1680000  231999  841000 1280000 1090000  449000\n",
            "  724000  126000  795000 2575000 1035000 1260000 8900000 1860000 4200000\n",
            " 5500000  430999 1151000  927999   51111  212000  428000  219000  749000\n",
            "  233000  614000   37500  865000]\n",
            "The unique values in km_driven are [ 70000  50000 100000  46000 141000 125000  25000  60000  78000  35000\n",
            "  24000   5000  33000  28000  59000   4500 175900  14500  15000  33800\n",
            " 130400  80000  10000 119000  75800  40000  74000  64000 120000  79000\n",
            "  18500  10200  29000  90000  73300  92000  66764 350000 230000  31000\n",
            "  39000 166000 110000  54000  63000  76000  11958  20000   9000   6500\n",
            "  58000  62200  34000  53000  49000  63500   9800  13000  21000  29173\n",
            "  48000  30000  87000  16000  79350  81000   3600  14272  49213  57000\n",
            "   3240 114000  53772 140000 175000  36000 155500  23000  22155  78380\n",
            " 150000  80362  55000   1136  43000   2650 115962  65000  56000 213000\n",
            " 139000 160000 163000  32000  52000  11240  66000  26500  72000  44000\n",
            " 130000 195000 155000   4000  41000  10832  14681  51000 200000  19600\n",
            "  46730  21170 167223 141440 212814  88635 149674   8000  68000  38000\n",
            "  75000  98000  81925  82080  97000  52047  62009  33100 220000  45000\n",
            " 180000  22000  80577 127500  40903  22288  61690  64484  75976  85962\n",
            "  57035  72104 164000 124439  77000   1250  17152  24005 149000  19000\n",
            " 109000  61000  27633  12586  38083  55328  81632 155201  93283 217871\n",
            "  90165 101504  86017  85036  91086 160254 125531  82000  84000 560000\n",
            "  14365  61083  66363  11700   7104  45974  55340  61585  39415  29654\n",
            "  64672  54634  66521  23974   1000  86000  52600  19890  11918  10510\n",
            "  47162  49824  58500  56580  46507  11451 172000  66508  29900   3000\n",
            "  85000   7900  17500 206500  88600 186000  11000 138000  27974  18000\n",
            "   1400 124000  42000  28205  32670  30093  56228  59319  39503  35299\n",
            "  51687  76259  44049  45087  41125  42215  54206  52547  59110  54565\n",
            "  47564  45143  61624 132000  10980  20629  69782  63654  59385  70378\n",
            "  55425  78413  40890  34823  55545  56541  43700  27483  56207   1440\n",
            "  91195  63657  97248  89000  12000  12997  26430  24600  28481  41988\n",
            "  30375   7658  34400  28942  53600  53652 106000 205000  79500 197000\n",
            "   9161  19077 128000  21302  10500 107000  55300  74300  48781  87620\n",
            "  40219  11473   8352   9745   9748  20694  31080  37605  55850  58850\n",
            "  23839  45454  46957 190000   1500  47000 116000  26350  71042 167870\n",
            " 133564  23038  43608  11212  49217  28838 135000  19571  29600  13500\n",
            "  48600 127643 102354  62237  21394  32686   1001  53261  14000  39895\n",
            "  73000  17000  18591  26766 300000  27620 223000 161327   6000  71000\n",
            " 144000  37000  26000  27000  13250 101000   8500  90246  60400  70950\n",
            "   1100  31491 107143  46412 107500  43826  55838 112880  30300  80659\n",
            "  81324 127884  66755 123084 806599  95851 234000 170000  96000  19014\n",
            "  23262  35925  40771  30500  55800  66569  81358  82695  68293 190621\n",
            "  64700  88470 126000  74183      1 192000  83411  13270  88000   7000\n",
            "  13770 102000 143000 115000 136906 133000  28689  80322  61658 185000\n",
            "  30600 235000  67000  74500 118700 223660   2000  73756  16400  41395\n",
            "  71014 181000  89550 149500  83000  44800 156000 146000  99000  37516\n",
            "  25880 136000   2020  94000  88500  52536   1950 118400   6480  32077\n",
            "  19107  18469  28217  72787  31063  79641  58692  54784  64156   9500\n",
            "  81366 244000 312000 145000   7300  72539    101  52328  91505  20500\n",
            " 154000  41723  68745  27289  24662  28245  27005  39227  31367  35008\n",
            " 100005  45264  39093  45241   2769  43128  22255  59213   1010   1111\n",
            "  48965   5166  76290  45766  78771  79357  76736  92645 101849 155836\n",
            "  63230   1758   1452  35122  92621  92198 152000  78322  54309  34600\n",
            "  38217  77073  16584  81257   3917  69069  59059  39039  33033  55168\n",
            "  41041  67067  66066  82082  70070  63063   9528 135200  50300 151624\n",
            "  74820 129000  66778  63400 157000  38500 103921  14825  43377 102307\n",
            " 245244  68500   5007  49600  43100  10171  41123  20118  52517  99117\n",
            "   3700  43500 137250   5400  11200  93000  62000   5800 267000 250000\n",
            "  28635  32114  95149  68458 105546 104000 132343  26134  52895  42324\n",
            "  60236  10300 142000  28643   7600  47253   4432  68523  80251  34500\n",
            "  42743  93900  55766 113600 138925 121764 105429  23122  44500  13599\n",
            "   5200  12700  95000  45839  74510  87293 156040  93415 101159  68519\n",
            "  55130  65239  58182  91245 102989 108000 178000  75118   4637  42655\n",
            "  69000 117000 105000 182000  24585  13900  17563 173000 151000 117780\n",
            "  81595   9700 221000  28740  48500 148620 270000  41090 296823  89255\n",
            " 168000   5550   1700  45217  44440  91365  90010  31800  59100  31200\n",
            "  22700  50900   2417  65500 140300  10211 260000  32933  54551  57112\n",
            "  41025  53122  64111  78892  74113  84775  20778  64441  43192  44416\n",
            "  79991  62601  89600  60800  69111  20969  20194  34982  44588  57904\n",
            "  59258  60826   1300  31707 115992 109052  90658  25552  40700  11174\n",
            "  72500  76600  97700  37500  23800  44077 210000   9422 240000  17100\n",
            " 224642 222435 159000 101100   1200 134444 238000 165000  63700  74800\n",
            "  60516  76731  63840  76400  31489 295000 158000 400000  19495  62668\n",
            "  85710  63356 129627   4400  14987  25061  42494  44875  89741 347089\n",
            " 222252  55250  12500 162000  22038   2500  89126 134000  42500 131365\n",
            "  48980  98900  13800  99700  49654  45457  39221  48220  11114  60208\n",
            "  98600  85441  64541  16267  71500  12999  14700  92686  49359 108731\n",
            "  29976  30646  23600  71318  78098  18054  38406  54350  32260  58231\n",
            "  59858  73350  88473  96987  77350  61187  68350  81150 280000 105700\n",
            "  37091  38900   9400  14100  37555  56600  67580  48238  38365  23670\n",
            "  49834  57353  68308  63240  64916  37161 118000  50852  53500  51500\n",
            "  79800   6590  49957  43235  50699 140730 256000 218000  66782 112198]\n",
            "The unique values in fuel are ['Petrol' 'Diesel' 'CNG' 'LPG' 'Electric']\n",
            "The unique values in seller_type are ['Individual' 'Dealer' 'Trustmark Dealer']\n",
            "The unique values in transmission are ['Manual' 'Automatic']\n",
            "The unique values in owner are ['First Owner' 'Second Owner' 'Fourth & Above Owner' 'Third Owner'\n",
            " 'Test Drive Car']\n"
          ]
        }
      ],
      "source": [
        "# Check Unique Values for each variable.\n",
        "\n",
        "for col in df.columns:\n",
        "  print(f'The unique values in {col} are {df[col].unique()}' )"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aJVzjfvx2MVy"
      },
      "source": [
        "## ***3.Data Wrangling and visualization***"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "C5zjfeGMKnVA",
        "outputId": "266914b9-aca0-4171-8b3c-3e4212e1d64b"
      },
      "outputs": [
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjwAAAI+CAYAAAC4x9CRAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAAsTAAALEwEAmpwYAABhT0lEQVR4nO3dd3zU9eHH8dfnLoMd9lCUw42KgjhQFLRaW42tHbZ2WK21btzr0NZebW1jd2vde9RZt6c/tSoBE5a4cCGiYSQkzBzZyd19fn/cgQEChOSSz/fu3s/HIw/C5cY7EZN3Pp/P9/Mx1lpEREREMpnPdQARERGRrqbCIyIiIhlPhUdEREQyngqPiIiIZDwVHhEREcl4KjwiIiKS8VR4REREJOOp8IjIFowxPzHGvG2MqTXGrDDGvGyMOTLFr2GNMXuk8jlFRLZGhUdENmGMuRz4B/AHYBiwK3ArcPIOPk9OysOJiHSQCo+IbGSMKQBuAC601j5tra2z1rZYa1+w1l5ljDnUGDPLGFOdHPn5tzEmr9XjrTHmQmPMImDRNl5nRvLd95OjSKcaYz40xnyr1X1yjTGrjTHjjTGB5HOfY4ypSL72la3u6zPGBI0xi40xa4wxTxhjBqb+KyQi6UqFR0RaOxzoATyzlY/HgMuAwcn7HgtcsNl9vgMcBuy7tRex1k5OvnugtbaPtfZx4EHgtFZ3OxFYYa19t9VtxwB7AscD1xhjjkveflHydacAOwHrgFu29voikn1UeESktUHAamtttK0PWmvnW2tnW2uj1toy4A4SJaO1P1pr11prG3bwtR8GTjTG9Ev+/WfAQ5vd57fJUacFwH3Aj5O3nwdcZ61dbq1tAkLAKZpWE5ENVHhEpLU1wOCtFQVjzF7GmBeNMZXGmPUk1vkM3uxuyzrywtbaCqAE+L4xpj9wAvCfbTz3EhKjOQCjgGeSU23VwCckRqOGdSSLiGQeFR4RaW0W0ERieqgttwGfAntaa/sB1wJms/vYTrz+AySmtX4AzLLWlm/28V1avb8rUJF8fxlwgrW2f6u3Hm08XkSylAqPiGxkrY0A1wO3GGO+Y4zplVw8fIIx5k9AX2A9UGuM2Qc4vxMvVwXsttltzwIHAZeQWNOzuV8nM+0HnAk8nrz9duBGY8woAGPMEGPMDl1VJiKZTYVHRDZhrf0rcDnwK2AVidGTqSTKyJXAT4Aa4C6+KhwdEQIeSE5D/TD52g3AU8Bo4Ok2HlMMfA68DvzFWvtq8vZ/As8DrxpjaoDZJBZOi4gAYKztzOiziEhqGWOuB/ay1p7W6rYA8CWQu7UF1SIi26IrGETEM5J755xF4gotEZGU0ZSWiHQJY8xRyU0Ft3jbyv3PJjF99rK1dkZb9xER6ShNaYmIiEjG0wiPiIiIZDwVHhEREcl4KjwiIiKS8VR4REREJOOp8IiIiEjGU+ERERGRjKfCIyIiIhlPhUdEREQyngqPiIiIZDwVHhEREcl4KjwiIiKS8VR4REREJOOp8IiIiEjGU+ERERGRjKfCIyIiIhlPhUdEREQyngqPiIiIZDwVHhEREcl4KjwiIiKS8VR4REREJOPluA4gIukjEAz3AAYCvXfgrSfgTz6FSb5teD8ORJNvsVbvNwDrW71F2vp7WVFhtMs+WRHJKMZa6zqDiDgUCIZzgaHA8OTbsG28X+Ao5tZsKEbrgBVAeRtvFcAKlSOR7KbCI5IFAsFwAbAnsEerP/cAdidRdszWH50R4sBKNi1Ci4FPk29lZUWFMXfxRKSrqfCIZIjkdNP+wF5sWmr2BAY5jJYOmoDPgYUkCtDGP8uKCiMug4lIaqjwiKShQDDcHxgHjAcOSv65D1+tlZHUqSRRfhYA84C3SRShuNNUIrJDVHhEPC4QDI8gUWg2vB0EjHYaSmqAd0iUn3nA22VFhYvdRhKRbVHhEfGQQDDsA8YCk5Nvk4ARTkNJe60F5vNVCSotKyqschtJRDZQ4RFxKBAM5wAT2LTgDHAaSlLpE+AN4E1gellR4RrHeUSylgqPSDcKBMP5wGEkys0U4HASe9VI5rPAB3xVgIrLigrXu40kkj1UeES6WCAYHgUUAicBxwA93CYSj4iRWAe0oQDNKCsqbHAbSSRzqfCIpFggGPaTGLk5iUTR2d9tIkkT9cBrwHPAC2VFhasd5xHJKCo8IikQCIYHACeQKDjfJHH8gkhHxYFSEuXnubKiwkWO84ikPRUekQ4KBMO7Aj8ETiYxoqM9cKSrfEKy/ABzyooK9Y1bZAep8IjsgEAwPAT4AfBjEldUZfqRDOI9lcALwKMkrvzSN3GRdlDhEdmOQDDcD/guiZJzLJDjNpHIRsuAR4CHyooKP3IdRsTLVHhE2pA8l6qQRMkpRFdWife9CzwEPFpWVFjpOoyI16jwiLQSCIaPBs4EvgP0cxpGpGNiwP9IlJ9nyooK6x3nEfEEFR7JeoFgeBBwBnAOsLfjOCKpVAs8A9xXVlT4puswIi6p8EjWCgTDRwLnAacA+Y7jiHS1T4BbgQe1w7NkIxUeySqBYLg/cDqJ0Zz93KYRcaKWxHTXLVroLNlEhUeyQiAYngicC5wK9HQcR8QrioFbSKz1iboOI9KVVHgkYyWPePghcCVwkOM4Il5WAdwJ3FlWVLjCdRiRrqDCIxknEAz3As4CLgcCbtOIpJUW4GngT2VFhe+4DiOSSio8kjECwfBg4CLgQmCQ4zgi6e4V4A9lRYUzXAcRSQUVHkl7gWB4N+AKEvvnaH2OSGqVAH8sKyoMuw4i0hkqPJK2AsHwBOBq4Pvo4E6RrvYeUAQ8WVZUGHecRWSHqfBI2klecfU74DjXWUSy0CLgJhLndzW7DiPSXio8kjYCwfCBwO+Bk1xnERGWA38Bbi8rKmxyHUZke1R4xPMCwfBewA0kLjE3juOIyKaWAr8hsYOzprrEs1R4xLMCwfCuJL6RnoHW6Ih43YfAtWVFhS+4DiLSFhUe8ZxAMDwM+BWJ4x/yHMcRkR1TAlxTVlRY4jqISGsqPOIZgWB4AHANib10ejmOIyKd8zwwrayo8GPXQURAhUc8IHkExPkk1ukMcBxHRFInBjwI/KasqHCZ6zCS3VR4xKlAMHwM8E9grOssItJlGoF/Ab8vKyqscR1GspMKjziRXJD8V+AU11lEpNtUAFeWFRU+6jqIZB8VHulWgWC4J4l1OlejYyBEstWbwFSt75HupMIj3SYQDP8A+DMwynUWEXGuhcR09m/LigprXYeRzKfCI10uEAyPJTF/f7TjKCLiPeUkprkecx1EMpsKj3SZ5PTV74BL0caBIrJtb5CY5vrEdRDJTCo80iUCwfAU4G5gD9dZRCRttAD/IDHNVec4i2QYFR5JqUAw3A/4E4ldknXulYh0xBfAL8qKCotdB5HMocIjKRMIhk8E7gBGus4iImnPAv8GgmVFhfWuw0j6U+GRTgsEw4NIXG3xU9dZRCTjLCYx2jPDdRBJbyo80imBYPiHwM3AUNdZRCRjWRLfZ6ZptEc6SoVHOiQQDI8AbgW+4ziKiGSPz0mM9sx0HUTSjwqP7LBAMPxt4F5gkOssIpJ14iRGe67VaI/sCBUeabdAMNwD+AtwoessIpL1PgdOLysqnOU6iKQHFR5pl0AwvB/wKDrVXES8IwpcDxSVFRXqh5lskwqPbFcgGD6fxMnmOuxTRLzoFRKjPStdBxHvysrCY4yJAQuAXBK/ITwI/N1aGzfGHAycbq29OIWvVwYcbK1dnarn7A7Jy83vAU52nUVEZDtWAD8tKyp803UQ8aZsLTy11to+yfeHAo8AJdba33TR65WRZoUnEAwfDTwM7Ow4iohIe8WBG0kcTRFzHUa8xec6gGvW2pUkjkGYahKONsa8CGCM6W2MudcYM9cY864x5uTk7fslb3vPGPOBMWbP5O2ntbr9DmNM2h2YGQiGcwLB8I3A66jsiEh68QG/Bl4PBMM7uQ4j3pL1hQfAWvsFidO8N9887zrgDWvtocAxwJ+NMb2B84B/WmvHAQcDy40xY4BTgUnJ22Ok2c7DgWB4ODAduBb92xCR9DUFeC8QDJ/gOsiOMsbEkr80b3gLJm+fnlxysaPPN84Yc+I2Pn6wMeZfncmcLnJcB/C444FvG2OuTP69B7ArMAu4zhgzEnjaWrvIGHMsMAGYZ4yBxALftFlAFwiGJwJPAfqtSEQywRAgHAiG/0Jiz56o60Dt1JD8pTlVxpH4xfylzT9gjMmx1r4NvJ3C1/Ms/RYPGGN2IzEis3lBMcD3rbXjkm+7Wms/sdY+AnwbaABeMsZ8LXnfB1rdd29rbag7P4+OCgTDZwHFqOyISGYxwFUkprgGuw6TKsaY440xs4wx7xhjnjTGbFiTeogxptQY835yeUUBcANwanK06FRjTMgY85AxpgR4aLNlHH2MMfcZYxYkl2t83+GnmXJZX3iMMUOA24F/2y1XcL8CXGSSQzbGmPHJP3cDvrDW/gt4DjiAxJqXU5KLoDHGDDTGjOqmT6NDAsFwbiAYvhW4G8hznUdEpItMBuYFguEDXAdph56tprNixphTW3/QGPMn4AWgN4nv23XA5caYPCBMYmQrh8Qsw9+AIuDx5C/ijyefZl/gOGvtjzd77V8DEWvtWGvtAcAbXfQ5OpGthWfDP6iPgP8BrwK/beN+vyNx6foHyfv+Lnn7D4EPjTHvAfsDD1prPwZ+BbxqjPkAeA0Y0bWfRscl1+u8CZzvOouISDcIAKWBYPh7roNsR8OGmYLk+49v9vGRJLZTiZNYe3oaMAr4BdALONpaux+JqawSoF8br/G8tbahjduPA27Z8Bdr7brOfSrekpWXpWe7QDB8GIn1OroKS0SyjSXxC+4NXtydebNtU1q/Px24ErgcGGOt3TDjUEliluFlIM9aO3az5/s5iW1Rpib/HgJqrbV/Sf79aOBKa+1Jxpj5wI+stYu6+NN0IltHeLJWq/U6Kjsiko0MEAKeCATDvRxn6YjlQMAYs4cx5jASBa4/MBroY4w5BMAY09cYkwPUAH3b+dyv0eqsRGPMgFQGd02FJ0tstl4n33UeERHHTgFKAsHwrq6DbGbjGp7k+0WbfbyexJTWByTW2NQB+yQ/diZwszFmIVAJfAYUAPtuWLS8ndf+PTDAGPOhMeZ9EtuxZAxNaWWBQDDcj8QU1nGus4iIeMxK4PtlRYVvuQ6yudZTWq1uC9FqSqrV7TOB6621b7a67d/A29ba+7shrudphCfDJXcbnYHKjohIW4aSuGz9l66DdNIfgb8k94fbQAc+t6LCk8ECwfC+JDZJPNB1FhERD8sD7goEw793HWQzvYwxy1u9Xb61O1prXwL+BbxsjPnYGFNKYn+5V7orrNdpSitDBYLhKcCzJBaziYhI+9wNnKfDRzOPRngyUCAYPpVEq+/vOIqISLr5JfBUIBjWdFCGUeHJMIFg+ArgUXQllohIR50MvBoIhvu7DrI9xpjhxpjHjDGLjTHzjTEvGWP2MsZYY8xFre737+SePBv+frkx5tPkMRLvG2P+ZozJdfJJdBMVngwRCIZ9gWD4H8BfSOwzISIiHXckMDMQDHt2z7LksUfPANOttbtbaycA04BhJK4+uyR55MTmjzuPxOHYE5MbFR6SvH9Gj2qp8GSAQDDcA3gcuMR1FhGRDLI/ieMo9tnuPd04Bmix1t6+4QZr7fvAMmAViTMez2jjcdcB51trq5OPabbWFllr13d9ZHdUeNJcIBjuA/wfiU20REQktXYF3koeyeM1+wPzt/Hxm4ArjTH+DTcYY/oBfay1X3Z1OK9R4Uljyfnl/wFTHEcREclkg4A3AsHwCa6D7Ahr7RfAHOAnW7uPMeYbyV2Yy4wxR3Rfuu6nwpOmAsHwYBLbinvxtw4RkUzTC3g+eRWsV3wETNjOff4AXENybWdy2qrWGDM6+fdXkiezf0hiP6KMpcKThgLB8AgSB4COd51FRCSL5AD/CQTDP3UdJOkNIN8Yc86GG4wxBwC7bPi7tfZT4GPgW60e90fgNmNM/+RjDNCjOwK7pMKTZgLB8EgSZWdf11lERLKQH3gwEAyf7jqITewc/F3guORl6R+RKDOVm931RqD1kRO3kVjQPMcY8wFQArybfMtY2mk5jSRP9X0T2M11FhGRLBcHzi4rKrzXdRBpH43wpIlAMDwKmI7KjoiIF/iAu795qWemt2Q7VHjSQCAYDpAoO6PdJhERkQ3GNcRnXzv3vss/2WfM2a6zyPZpSsvjWpWdUW6TiIjIBuMa4rMun3d//pC1Hx8EWOCXYz79RNNbHqbC42HJq7HeQtNYIiKekSw7PYas/bj1lbJx4Iwxn37ysKtcsm0qPB4VCIYHkLgaa6zrLCIikjCuIVZ6xdz7ew1e98m4Nj4cA3405tNP/tvNsaQdtIbHgwLBcC8gjMqOiIhnbKfsQOKS9Uc+2WdMYTfGknZS4fGYQDCcR+L028NdZxERkYRE2blvW2Vng1zgyU/2GaMjfzxGhcdDAsGwD3gYON51FhERSfiq7Hw6rj33j0N96Ce+P499YOxBXRxNdoAKj7fcDvzAdQgREUkYVx8tvWLufb3bW3ZihhXX/MJf/fEo3yHA/419YOxeXZtQ2kuLlj0iEAzfBFztOoeIiCSMq4+WXjHv/t6D1316YHvuH/Wx5LJz/P6qAab1MQ5LgEkLzlhQ3jUppb1UeDwgEAxfDdzkOoeIiCSMq4+WXjn3vt6Dqhe2q+w05fDZRef7B1T3MUPa+PAC4KgFZyyIpDal7AgVHscCwfAvgbtc5xARkYQdLTt1+SyYer5/17qepmAbd3sT+OaCMxY0pyal7CgVHocCwfAJwAskLmUUERHHEmXn3j6Dqj87oD33X9ebty86379vc67p1Y67Pwb8ZMEZC/SD1wEtWnYkEAzvR+Ifv8qOiIgHjKtv2aGys2IAsy680H9AO8sOwI+AP3Y8oXSGRngcCATDQ4C5QMBxFBERYWPZ6TuoelG7NnxdPJyZ1/7cP8ka05GBg/MWnLHgjg48TjpBhaebBYLhfOAN4AjXWUREZMfLzvujTfGNP/J3ZmPBGFC44IwFr3TiOWQHaUqr+92Nyo6IiCeMq28pvWrOPe0uO8X7m+mdLDuQWMrw+NgHxo7p5PPIDlDh6UaBYPg64DTXOUREBMbVJcrOwMjn2y07FuLPH2Zm3PIt/9EpevkC4IWxD4wdlKLnk+1Q4ekmgWD4FOB3rnOIiEiy7My9u187y070oa/5Zj/8Nf/kVGYYHI2tK16y/H5CBTmpfF5pmwpPNwgEwxOABwDjOouISLb7quws3n9797XQcOtJvndfPMyX0qUIhzQ0Fv9vWfn4gfH4ScA/U/nc0jYtWu5igWB4ZxJXZO3kOouISLYbV9dSevXcu/sNaF/ZWX/TKb4v39nT164NCNvF2pazI+tnX7wuctRmHzmXUOTOlL2ObEGFpwsFguEewFvABNdZRESy3fjkyE57yk4c1vzmNP/KhbuYlC0sNtau/dfK1UuPrm8Y18aHW4BjCUVmpur1ZFOa0upa/0JlR0TEufF1LaVXz7mroD1lJ2aouPosfySVZSc/Hv/8+eUrardSdgBygacIFeyaqteUTWmEp4sEguGfAQ+6ziEiku02lJ3+67/Yb3v3bfFRdtm5/tyV/c3OqXr9YdHovGeXr9inj7V923H3OcBRhCItqXp9SdAITxdIHhtxu+scIiLZLlF27mxX2WnMZeEFF/p7p7LsHFHfUPzqsooJ7Sw7AIcBN6Xq9eUrGuFJsUAw3AeYB+zjOouISDYbX9ecHNn5crtlp7YHH0w93z+qvsc2TzxvP2ubL6yOzD2vev2RHXyGkwlFnk9JFgE0wtMV7kRlR0TEqUTZubN/e8rO2j68fd5U/x6pKjvG2lW3Va36tBNlB+B+QgWBVOSRBI3wpFAgGD4fuNV1DhGRbPZV2Snbd3v3rRjArCvO9h8c85vcVLx2j3j8s6fLK3vtEo2OTMHTzQWO1Hqe1NAIT4okNxf8u+scIiLZbHxdc+k1s+8c0J6y8/kIZl52rv+wVJWdES3ROdOXlu+UorIDcCjwpxQ9V9bTCE8KBILh/sA7wGjHUUREstb4uqbSa2bfNaCgpmy7l5O/u5uZ/sdTU3YuFlPq66ffXLV6iumaHfW/QyjyXBc8b1bRCE9q3I/KjoiIM4myc2f/9pSd6WNNccrKjrWNl61dV/LvqtVHd1HZAa3nSQmN8HRSIBi+BPiH6xwiItnqoNrG0qvn3NW/oGbJNqexLMSfm2jeeuSY1BwC6rN25Z2VK1cd1ti03YXRKTCPxHqe5m54rYykEZ5OCATDY4Ai1zlERLJVouzcOaAdZaflwWN9s1NVdnrG45++vLwi1k1lB+AQ9POmUzTC00GBYDgHmI2OjhARcWJ8bWPpNXPuHFBQs3Sb01gWGm45yffhjLG+Q1Lxuru0tMx6qrzywJ7W9krF8+0AS+K8rTe7+XUzgkZ4Ou7XqOyIiDgxvrax9JrZd7Sn7ET++EPfolSVnePq6qeHl6+Y6KDsQGKN0L2ECtq7a7O0osLTAYFg+GDgWtc5RESy0fjaxtLg7DsGFtQu22bZiRtW/fpn/hXv7e47oNMvam3D1WvWzfr7yi5dnNweAeCvDl8/bWlKawcFguEewLtoN2URkW63oez0q122ze/BMUPFVWf5m5YPMZ2+gtZn7Yp7V6ysntDUlLLT01PgBEKR/3MdIp1ohGfHFaGyIyLS7RJl5/btlp0WP19efJ6fVJSd3vH4x68sqzAeKzsA9xAqGOA6RDpR4dkBgWD4GOBi1zlERLLNV2Vn+TbLTmMun15wob/vqv5mp86+5ujmltLipct3Gx6LDe/sc3WBnYB/uQ6RTjSl1U6BYLgfsADY1XUWEZFsMr62oTQ4+45B/WqX772t+9X04P2LzvcHOn0IqLX2xLr6GTetWjOlU8/TPb5HKPKM6xDpQCM87fcvVHZERLpVe8vOmj7MO3+qf68UlJ26X61ZNydNyg7A7YQKBrsOkQ5UeNohEAx/CzjDdQ4RkWwyvrahNDjr9u2WnfKBlE69wD+uOdf07Mzr+a2teHhF1fJTa2onduZ5utlQ4HbXIdKBprS2IxAM9wY+RqM7IiLd5qDa+tLgrDsG960r32tb9/tsJ2b8+nT/kdaYTv0C3ycWX/B8ecXwIbH4kM48j0M/IRR51HUIL9MIz/aFUNkREek2ibJz+3bLzvw9TPGvzsiZ3Nmys0dzc0nx0uV7pXHZAfgHoYL+rkN4mQrPNgSC4QOAS13nEBHJFl+VnYptlp03DjDTb/qBv3PrbKyNf6emtviZ8spJeZDfqedybyjwB9chvExTWlsRCIYNUAIc7jqLiEg2aE/ZsRB/5ghT8tgU/1GdejFra29Yvfbj79bWHdqp5/GWOHAYocjbroN4kUZ4tu5sVHZERLpFouzcNmQ7Zafl/uN8czpbdvzWLn+0ompFhpUdSPxMv41QgX62t0FflDYEguEhJHZUFhGRLvZV2Vmx59buY6H+5m/73n/5EF+nfhHtF4u9/8bS8p77Nzdv9bXS3MHAea5DeJEKT9v+CmjLbhGRLnZQbX1psHS7ZSfyhx/6Pn9rP9/BnXmtMU3NM99cWj5mYDw+qDPPkwZuJFQwzHUIr1Hh2UwgGD4a+JnrHCIime6gmrrSaaW3De1bv/WyEzes+tXp/sr3O3PiubWxU9fXFD9RUXlUHuR1+HnSR3/gL65DeI0WLbcSCIbzgPfR4aAiIl3qoJq60mmzbhvap75yj63dJ2Yov+osf8vyISbQ4Reydv0fV6357KS6+k6NDqWpowlFil2H8AqN8GzqalR2RES6VHvKToufLy463+/rTNnJsXbJkxWVq7K07ADcSqgg13UIr1DhSQoEwzsD17rOISKSySa0o+w05PLJ+Rf6C1YXmBEdfZ0Bsdi7by4t77dPc8vuHX2ODLAvcLnrEF6hwvOV3wOdOodFRES27qCaupLgrFuHbavs1PTkvfOm+nde39t0eGHx2MamGa8vLd+/fzyui0/g14QKhrsO4QUqPGzcUfl01zlERDLVQTV1JdNm3Tq8T33VVkdcVvdl7nlT/fs09DD9OvQi1sZOi6wvfmRF1eRc0FROQm/gt65DeIEKT8Jf0NdCRKRLHFRTV3Jt6S0jtlV2lg+iZOoF/oNackyPDr2ItZE/r1rz3jVrqzt33ERmOotQQdavT836H/KBYPgbwNdd5xARyUQT1teWXlt6y4jeDSt329p9Fu7MjCvO9h8e95mcjrxGrrVfPlNeufabdfUTOp40o/mBm1yHcC2rL0sPBMM+4D1grOMoIiIZZ8L62tJps24dvq2yM29PM/3Pp/iP7uhrDIrG5j9XXrFHQdwWdPQ5sshRhCJvuQ7hSraP8PwclR0RkZSbsL6mdNqsW7c5svO/A01xZ8rOhIbG4teXlY9T2Wm3P7sO4FLWjvAEguFewCJgJ9dZREQySaLs3Daid8PK0W193EL8qUmm5InJHTwE1NqWX0TWz7psXWRyp4Jmpx8QivzXdQgXsnmE5wpUdkREUio5sjN8G2Wn+d6v++Z0tOwYa9f9c+Xqj1R2OuwP2boZYVYWnkAwPIzErsoiIpIiE9bXlCSmsVa1OY1lof6fJ/sWvHJwx048z4vbxc+Vr1j/tfqGcZ0Kmt32BM5xHcKFrCw8QAjo4zqEiEimmLC+pmRa6S07925YtbWRnciNp/oWl+7r69CVVEOi0benL10+ZHRLdFTnkgpwPaGCvq5DdLesKzyBYHhX4CzXOUREMsXGstO4OtDWx+OGVdee4a/6YDdfhy4SmdjQWPy/ZRUH9bW2YxsSyuaGAle5DtHdsq7wANPQDpwiIikxYf360m2VnZiP5Vf80l+/eCez1w4/ubXN562LzLyrcuUUX3b+vOpKlxIqyKqjN7LqH1AgGB4J/MJ1DhGRTDBh/frSa7dRdpr9LJ56nt9fPtjs8DSUsXb1LVWrPrmwOtKxK7lke/oCl7gO0Z2y6rL0QDD8L+Ai1zmkfWw8xooHLiOn7yCGnvIbGpa8T/Wb92JjLeQN34NBJ1yC8fm3eFx0/UrWvHwz0fWrMMYw9AchcgqGseqFP9Oyagk9dz+EAVPOAKC69DHyBo+i114dWkMpkrUOXh8pnVZ66869Gte0WWYa8vj4ovP8wzpyCGh+PL7o6fLKnrtGoyM7n1S2YR0wilCkxnWQ7tChbbzTUSAYHg6c7TqHtF/N28+TO2gXbHM91sZZE/47w350I7kDd6Z65sPULnidvgcev8XjVr/4NwoOP5Weo8cTb24AY2he+SW+nHx2+sW/qXrsV8Sb6oi3NNFcsZD+R/zIwWcn4lbDF/NZ+/qdEI/T58DjKZj4g00+bqMtrA7/jebKz/H17MuQk68hp2AYjcs/pvbFv9Z/0FRz8MoRw/MCeXmsj8W4vKKcO0fugs8Y1vfkvYvO9+/ekG92eGHsiGh07jPLV+zb21pdWNL1BgAXAkWug3SHbJrSuhro2KF00u2i61fT8MU8+iQLTbyhBuPPIXfgzgD0CIyj/rOSLR7XvHopxOP0HD0eAF9eT3y5PTC+HOLRJqyNY+NRMD4iMx+m4Mifdt8nJeIRNh5j7Wu3MfQHv2WnX95K3cfFif93Wqn94FV8PXqz87l30e/gk1k3/f7EY6c/sPbRnUauvX7I4LzHq9cBcPuaNZwzaBA+Y1jdj7nnT/Xv05Gyc1R9Q/H/Las4WGWnW11OqKCX6xDdISsKTyAYHgKc6zqHtN+61++k/9G/wBgDgK9nP2w8RtOKRQDULywhtn71Fo+Lri3H16M3K5+5kYr7Lmbdm/di4zFyB++Cv2cBK+6/hF57HEp03QqsteQP36NbPy8RL2he8Rk5/UeQ2384jUs+IFa7lqr/XENk9pMb71O/aDZ99j+W2gX/Y81rt1O/sITVt51VV7Dmyz45jetGljU38eL69Zz4xRd82tTIob168+VAW3Ji7bJDmmJ2x365tLbp4rXVJbdWrdLi5O6XNT8fs2VK60ogKxpsJqj/fC6+3v3JH74HjUs/AMAYw5BvX826N+7CxlroETgIfFt+X7TxGI3LPmLEmf8ip98QVj9308apr4HHfbXX1sr//paB35hKpPRxmld+SY/AOPqO+2a3fY4iLkVr1pDTb8jGkZ5+E39AdF0FdR8X03OPw8gbvCux2jX4+w6hZW05ffadTMvC0sZndt5lXU396pHBFRWsjEa5fugwnl2/niYb55ORFE8dFplc0LO/8eW3v7MYa1fdWbmyamJj06Qu/JRl264kVHAroUiT6yBdKeObdCAYHgRc4DqHtF9T+cc0LJrD8tt+warn/0Tjkg9Y/cJfyN95DMN/+idGnP53euyyH7kDdt7isTl9B5M3bDdy+w/H+Pz03HMizVWLN7lP/aLZ5A3fA9vSSEv1CoZ8J0j9whLiLY3d9SmKeMKGkR5/rwKMz0fvMZNpWDR7i/v1r6le0b+5Pqd3c/XIMT168NioAIX9+rE82kKB38fKfLv25+vKp6wMrzJ99m//bFSPeHzhy8srWiY2Nu2fys9LdthOZMEVzBlfeIDL0a7KaWXAlJ8z8sIHGHn+vQz59tX0GHUAg791JbG6aiCxmHL9nP/SZ/wJWzw2b8SexBtridVHAGhc8gF5g3fZ+HEbi7L+7efod9j3sdEmwCQ/EIdYtKs/NRFPyOk7iOj6VRtHemI1q/H3GYS/72BitWsA8PcZRKxmFTtXr/68ZvGcoStbmnJClZWsaGkB4EcF/bljzRpm2caayl38A/19/Aw5cQhrX1/brgwjW6Kzi5eWj9w5GtOZht5wDaGCjJ71yejCEwiG+wNTXeeQ1Fg/92nK7zqPivum0nP3w+g56kAAmlYsYs3L/wLA+PwMOOYsqh67jop7LgQsfQ78xsbnqHknTJ/9j8WX24PcIaOx0SYq7rmQvOF74OuhXizZIW/EXkTXVRCrq8bG49R9MoOeexy2yX167XkYObOeqvpr9ac9g0MH+7/ety9H9OnNtZUrAJjb0BAff8DAL4bcvFdfG7NEI1HyhuQRmR9h6a1Laarc+uzIsXX1xS8trzisl7W9u/QTlR0xCviZ6xBdKaP34QkEw1cCf3adQ0TEaxoWz2PNK7cQb1hPweGnUnDEqVQ+ei0Asdo15DTWNw6PNfhj8Vhuf7+fv4zYiZ1yczls0SLG5Ofb91oabd7uPXy7nrcry+5aRsuqFlrWtpBTkMOIn45g/dvrGXn2SMr+WsaoS0bhy/eBtY1Xra1+5/T1NUc4/vSlbYuAfQhF4q6DdIWMHeEJBMN+NLojItKmnrsfws7n3YO/9wB67TsFG2sh3rCe6Lpyjjlu6vw3d9trdT4293fDhvP4qAC75OXxZm0tOQbyjyxYMuae/XzDvjOM5Xcvp+cuPek3oR/9JvRjxE9HsK54HfHmOGvfWEv/w/vjy/fhs7by3sqVX6jseNqewEmuQ3SVjB3hCQTD3wOecp1DRMTLGhbPY+3rd4GN02PUOPIqF1efFK/zHZpn+n3e1MxbdbWsi8XIMYY+Pl/0k7yW+Mjrd8vLG5RHPB7n419+zD7/2ofVL60mb1gea15ZQ8u6FnY5fxdWv7KawBUBemM/fm75ioHDY7Hhrj9f2a7/EYp83XWIrpCxIzzAxa4DiIh4Xc/dD2Hnc+5k53PvZu+CnT6d2LQuNziwX7+v9enL8Jwc9srP54XRu/HU6NErR16/25e5Y3rnrZ+/HoCad2ogOfkx8NiBVJdW4+vpY7drd6P2o1qGnDSEQCxaWry0fLTKTto4jlDBvq5DdIWMLDyBYPhAYIrrHCIi6eKQ6rUl3/1yxpCetmWLhcRRH8suP9vfsHiE2XP4qcOpX1jP59d/Tv3CenIG5GCMIW9QHrtN243df707Js/QsraFb/SOlQy4+Ysjzniyrudna2IuPi3pmIxcDpKRhQeN7oiItNvB1WtLps26NbCbaRlU2fLV9gyV0Si9c/xrLzrfn1sxKHHiee6AXHa9aFf2uGEPhn5/KAD+3pse4lv1ZFXsusn57w16tWrSL8fn8qfjevDb4oze0y7TnE6ooMB1iFTLuMITCIYHAz9xnUNEJB0cUr229NpZtwZ6NlXvvH+PHixpaWZ5czPN1vJsTaRxzrn9/Wv6mY3TUdGaKDaeWPu5+sXVDDhqwCbPV/dhzZrjTFPk6j5N4+pbwGcSb/Ut3ft5Saf0JgM3IszETYbORYeEiohs1yHVa0uvLb1lVI/myM4AOcZw3dBhnL18GQ3Y5pwTBvjie/YsqHq6ip6je9JvfD/qPq2j6r9VAPTeuzcjfjZi4/P1isU+6vff8j3v+mHPPIBzJuTy06cbiMbhtkJ9W04zFxIq+GcmXaKeUVdpBYLhHKAM2PLMARER2WjzstPaqn7MueRc/7hojslv7/Pt3txc8nhF5YR8q184M8i3CEVedB0iVTJtSusUVHZERLbpkOo1Wy07S4dQctH5/gntLjvW2m/X1E5/trxykspOxsmo9bCZVngy6j+OiEiqHVK9puTa0lsCbZWdj3eh+Mqz/EfEfaZ9yx2srf3N6rVzb1y99uhU5xRPOI5QwT6uQ6RKxhSeQDA8FjjcdQ4REa86pHp1ybWlt4zu0bx+iwM7Z+9tikOn5UzBGNOe5/Jbu/yRiqqKU2rrDtv+vSVNGTLoEvWMKTzAma4DiIh4VaLs3LpF2bFgX55giv/2PX+79y7rF4steH1peY+xzc17pT6peMxphAoyYqoyIwpPIBjOBU5znUNExIuSZWe3NspO7ImjfCX3Hd/+srN3U/Nbby4t33tQPD449UnFgwqA77gOkQoZUXiAE4EhrkOIiHjNoetWbSg7I1rfbqHprm/45j11pO/Idj2RtfFT1tcU/7ei8sg8yOuSsOJVP3cdIBUypfD83HUAERGvSZad3dsoO7V/+67vo/8d5JvYrieydv2Nq9fM/82adTqyJzt9nVDBFuu+0k3aF55AMDwEKHSdQ0TESzaUnfyWmk0O7bSw7nc/9i2Zs4/voPY8T461S5+oqFz17dr6Q7omqaQBHxmwbCTtCw/wUyDXdQgREa/YWtmJG6qm/dy/+sOAb7/2PE9BLPbeG0vL+4xpbtm9a5JKGjnDdYDOyoTC83PXAUREvCJRdm7ZouxEfSy97Gx/0xcjzJ7teZ79mppmvrm0fL8B8fjArkkqaWZfQgVpPcqX1oUnEAyPAw50nUNExAsOXbey5NrSW/bIb6ndpOw05/D51PP9+SsGmV23+yTWxn4SqSl+rKLqqFyNnsum0nqUJ60LD9p7R0QE2FB2bt0jv6V2WOvb6/P46LwL/YPW9jPDtvbYjayN/GnVmvemrdXiZGnTjwkVpO0VemlbeJJ77/zEdQ4REde2VnYivXjnvIv8o2p7mQHbe45ca8ueKq9ce0Jd/YSuSyppbiDwLdchOiptCw+JvXe08ZWIZLXD1la1WXaqCph9/oX+/RrzTJ/tPcfAWOydN5eW99+rpWV01yWVDJG201rpXHh+4DqAiIhLE9dWlU6bddsWZWfJUN665Dz/Ie058XxcY9OM15eWH1AQj/fvsqCSSU4gVJCWC9nTsvAkp7NOcp1DRMSViWurSoOlt+6+edn5aFdTfNUv/JPiPuPf5hNYG/159foZD62ompwD7TsdXSTxbyUtp7XS9R/5sSTO9xARyTqHra0qCZbeumd+tG5o69tLxpjp//yO/+jtPd5Yu+5vK1eXHVffMLnLQkom+x7wgOsQOyotR3iA77sOICLiwsQ1lSXTNis7G048b0/ZybP2i2fLV6w/rr5hfJcGlUx2PKGC3q5D7Ki0KzyBYNgPnOw6h4hId5u4prJk2qxb99qs7EQfm9y+E88HR2Nvv7l0+aDdWqKjujapZLgewAmuQ+yotCs8wFHoZHQRyTIbyk5etH7j9z8LTXec4Jv/zKTtn3h+SEPjjP8tKx/fL261HEBS4XuuA+yodCw8afdFFhHpjIlrVrRVdmr/+j3fx2+M8x22zQdb23LOusjMeytXTvbDthcyi7RfYbptQphWi5YDwbABvus6h4hId0mUnds2KTtxWHvDT/yVH48y21yHY6xdc3PVquVTGhqP6vqkkmX6kbiA6GXXQdor3UZ4DgVGug4hItIdJq6pKJk267a9W5edmKFy2pn+tR+PMvtu67H58fjnLyxfUT+loVHnDUpXSasBiHQrPJrOEpGskCg7t++dF63fuKN81MeSy87xt3w53OyxrccOi0bnTl9aPmxUNLpL1yeVLHYyoYK06RFpEzRJhUdEMt7E1eVblJ2mHBZdeIG/Z+VAs80SM6m+YfqryyoO7mNt365PKlluKDDJdYj2SpvCEwiGxwDb/K1GRCTdTVxdXjJt9u37tC47dfl8eP5U/5B1fc3QrT7Q2qap66rfur1q1dG+NPreLmkvbaa10ul/iuNdBxAR6UqHr16eLDsNgzbcVt2Ld86b6h9d29P039rjjLWrbq9a9dm51eu3e3m6SIqlzX486VR4vu46gIhIVzl89fKS4Ow7Nik7Vf2ZfcFU//5NeWaru9r2iMcXvrS8onlSQ+PY7kkqsol9CBXs7DpEe6RF4UkeFrrdXURFRNJRW2Xny2G8dcm5/kOifrPVvU52aonOKV5aPnJkNJYWP3AkY6XFgERaFB5gItDHdQgRkVQ7fPXy0mmzbh/TuuwsGGWKrzlz2yeeH11XP/3/llcc2svatDvTSDLOca4DtEe6bDyYFu1RRGRHJMvOPrmxxoEbbntrXzP9Xydv4xBQaxsvX1c9/8xIzdbvI9K9jnUdoD3SZYRHhUdEMsrhq5aVtC47Fmz4EFO8rbLjs7bqnsqVi8+M1KTNpcCSFYYTKvD8GjLPj/AEguH+wCGuc4iIpMrhq5aVTJt9x5hWZSf6yNG+Oc8d7tvqWsWe8fgnz5av6L9TNLZf9yUVabfjgAWuQ2xLOozwHIMOvBORDHHEqqWbl53G2070vfPc4b6tjtrs2tIyq3hp+aidorER3ZdUZId4fh1POhQeTWeJSEY4YtXSkuDsO1uXnZo/f9/36fQDfYe2+QBr7fF19cUvLl8xsae1vbo1rMiOmUKoINd1iG3x/JQWKjwikgEmrVxaes2cr8pOHNb+9qf+yk92NePafIC1DcG169776fpabckh6aA3cDgww3WQrfF04QkEwwF0nISIpLkjVi4tvWbOHWNyY00DAGKGFcEz/fVLhrV94rnP2hX3rlhZPaGp6fDuTSrSKcfh4cLj9Skt/WYjImlt0solJcFWZSfqY8ml5/hjS4aZ3du6f+94/KNXl1X4JjQ1jenepCKd5unL071eeCa6DiAi0lGTVi4puWbOnftuKDtNOXx24QX+nlUDzci27j+6uaW0eOny3YfFYsO6N6lIShxEqGCrO4O7psIjItIFJq1cUhKcfcfGslOXz4LzpvqHtXniubW2sLau+PnyFUfkW3p0e1iR1OgBHOg6xNZ4dg1PIBjuBXh+IyMRkc1NWllWEpx953458eb+AOt6M/+i8/1jmnPNlldaWVv36zXrFvywRouTJSMcBsxzHaItXh7hORjtvyMiaWZS1RelrctOZX9mXXihf2xbZcdvbfl/VlQt/2FNrUazJVN49t+yZ0d48PAXTUSkLZOqvigNzrl73w1l54vhzLz2DP8RbR0C2jcWX/B8ecXwwbH43t0eVKTrHOY6wNZ4eYTHs180EZHNJcvOxpGd90eb4uCZOUe1VXb2amp+a/rS5XsNjsWHdHtQka61B6GCQa5DtMXLhUcjPCKSFo6s+qIkWXYKAGbsZ6bf+CP/lmtyrI1/r6Z2+lMVlUfmQX63BxXpHp4csPDklFYgGN4F2Ml1DhGR7TmyanHJNXPu2T8n3lxgwb54qJn50LFtnHhubc3vVq/95Du1dVt+TCSzHAa85DrE5jxZeNDojoikgc3KTvThY3xzXpjom7z5/XKsXfZwRVXjfs3NbZ+ZJZJZPDnC49UpLU9+sURENkiUnbvHJstOw22FvndemLjliecFsdj7ry8t77Vfc/OeLnKKOHAooQLjOsTmvFp4NMIjIp51VNXnpYmy09LPwvqbTvF9Nv2ALU88H9PUPPPNpeX7DozHPbmIU6SLDAD2ch1ic54rPIFg2ADjXOcQEWnLUZWfl1495579c+It/eKw5jen+cvf2dO36e6y1sZ+tL6m+ImKyqNyIddRVBGXDnEdYHNeXMMTIHHMvIiIpxxV+Xnp1XMTZSdmWHHNL/wNS4eaTQ/5tDZStGrNosK6eu2cLNlsP9cBNufFwuO5L5KISKLs3D02Jx7t2+Kj7LJz/bkr+5vdWt8nx9olj5VXxvZuaTnYVU4Rj/Dcz3IVHhGR7Zhcuajkqrn3HJATj/ZtzGXhRef5B0b6mE02DRwQi737/PIVgf7x+ABXOUU8ZF/XATbnuTU8qPCIiIdMrvysdEPZqc3ng/Om+odvXnYObGya8cbS8rEqOyIbjSZU0NN1iNZUeEREtiJRdu4dmxOP9l3bh7fPu8i/R30PU7DxDtZGT4+sn/HwiqrJOd4cMRdxxQeM2e69upGn/gdNXqG1j+scIiKTKz8rvXrOvQf4bbRPxQBmXXG2/+CY32y84spYW/2Xlau/OL6+YYuNBkUESAxgvOM6xAaeKjzAaKCX6xAikt2mrFhYetXc+w7w22ifz0cw87oz/JOsMRtHxHOt/fLJ8hVm95boQS5zinicp9bxeK3waDpLRJyasuLT0qvm3n+A30b7vLubmf7HUzc9F2twNDb/2fKKPQritmArTyEiCZ76ma7CIyKS1LrsTB9rpt960qZl5+CGxuK7K1ce6Qe/o4gi6cRTP9O9tmjZU18cEckeibJz3wE+G+313EQzY5OyY23LL6sjM++rXDlFZUek3UYTKvDMMhWvjfB4ar5PRLLDlIpPS6+ad9+BPhvLe/BY3+zwoV+deG6sXfuPlauXfq2+4SiXGUXSkCFxpdZ810HAe4VHV2iJSLc6uuKT0ivn3X+gz8Z8t5zke2/GWN8RGz6WF7eLnypfkRuIRsc5jCiSzvZGhWdTgWB4MLpCS0S6UauyEyv6ge/zd/fwbTzwcGg0Ou/Z5Sv27mttP5cZRdLcKNcBNvBM4QF2dR1ARLLHhrJjbKzh1z/zr/5spNl44vnh9Q3Ft1etOsrnvXWOIunGMz/bVXhEJOscXf5R6ZVvPzgOYtVXneVvWjbUJKbTrW2+oDoy9/zq9TrpXCQ1PPOzXYVHRLLKhrIT98WqLjvHn7eyvxkNYKxdfWvVqoojGxqPdJ1RJIN45me7Co+IZI1jyj8qveLtB8e15MSWXnS+f3CktxkMkB+PL3qmvLLnLtHoAa4zimQYz6zh8dL8tAqPiHSZry3/sPSKtx8cV58fW3TeVP+IDWVnRDQ6t3hp+YhdotGRrjOKZKC+hAr6uw4BKjwikgW+tvzD0svnPzg+0jv28flT/XttOPF8cn1D8f8tqzi4t7V9XGcUyWCe+PmuwiMiGW1D2akcEH/3wgv8Bzbnmp5Y23Tp2uqSW6pWTdGVWCJdzhM/3z2xhicQDOcBw13nEJHMcszyBaWXz39o/Bcj4vN+dbr/SGuMz2ftyjsqV66c2Ng0yXU+kSyhwtPKSBJbUIuIpMTXln8w6/L5D49/fzc7t+iHOVMAesbjnz5TvqLfztHY/q7ziWQRFZ5WPPHFEJHMcOyy92dd9s5/xs0Ya+feVuifAjCypWX20+WVB/S0Vju6i3QvT/yM90rh2cV1ABHJDMcue3/W5fMfPuC5w3n70aMTZee4uvriv61cPdloJFnEBU9cAemVwjPYdQARSX/HLnu/9PL5D+//wHHmg5cP8R2FtQ1Xr61+92fra7Rzsog7g1wHAO8UngGuA4hIejt22Xull8//z5h/f9t89tZ+vsN91lbeU7ly7cGNTUds/9Ei0oUGug4A3ik8nvhiiEh6Om7Ze7Munf+f3Yt+aJa9t7vv4N7x+MfPLl8xcHgstq/rbCLijUENr+w/4Ykvhoikn+OWvTfrknf+s+v1p/uq39vdd0CguaV0+tLy0cNjMW11IeIN+YQKnF8s4JXCoxEeEdlhX1/67qyL3n1k2NVn+VoW7cReJ9TWFb9QvuKIHtb2dJ1NRDbh/Oe8V6a0NMIjIjvk60vfmXXBB48OuPQ8X8/V/Si4bs26OT+qqdXiZBFvGggsdxlAhUdE0s7Xl75Tes6Hj/a56ALf4NpeND+8omrZgU3NE13nEpGt0ghPkvMvhIikh68vfaf0zE8fzbvwQl/An2eXvrpsxdChsdjernOJyDY5/znvlTU8/V0HEBHv+/qS+aU/XfyYuegC3347+VoWFC9ZvufQWGyo61wisl3OZ3KcF55AMNwX74w0iYhHHb/k7VnfW/549NJzOOiExrq5z5ZXTsqDfNe5RKRdnI/weKFoOP8iiIi3Hb9k3qyvr36i8ZpfmAmhNeve/15tnRYni6QX5z/rvVB4nA9ziYh3Hb9k3qzD6p+ov/Gnvr0fqaiq2r+5+VDXmURkh/V3HcALhaev6wAi4k3HL5k3ax/fkw33nsyQ15eW9xgUj3viEEIR2WHOp5+9UHicfxFExHuOWzKvZHifJ6OzDonmvbm0cp88yHOdSUQ6zPn/vyo8IuI5X1s2t6TP0KfI2aOOJyvWHek6j4h0Wq7rACo8IuIpkyvmleTt/F+OG7I2/6Q19ZNc5xGRlNAIDyo8IpJ0ROW8kvzAk9FLelSNHFPXsrvrPCKSMhrhwQOtT0TcO3j1vJKBox+v/52pnDCgOe78ElYRSSkVHlR4RLLeAdXzSg7Y9eH10+Krjsu17r8xikjKOf9Z74XC43cdQETc2adu3sxTd7q3+afxyAmus4hIl3H+i4wKj4g4s2/z3LdCg+7oeWi84SjXWUSkS2mEBw+c5yUi3e+QeOlb/+pz204jiO3mOouIdDmN8KARHpGs8y2ml97U4679evmsjpYRyQ4qPGiERySrnOV/qfRXOQ8faownvv+ISPdQ4UGFRyRrXJbz5MyL/c9MMkb/34tkmZjrAF4oPM2uA4hI1/ttzv3Fp/tfnWwMxnUWEel2Ta4DeKHwNLoOICJd6++5txR/118yxXUOEXFGhQcVHpGMdk/un6cf63/3aNc5RMQpFR5UeEQylLVP5N0w41DfwqNdJxER51R48MAXQURSy0c89kLedbP28y3RNJaIgAd+1nuh8GiERySD5BBteTXv6rd381Ue6TqLiHiGCg8qPCIZI5/mxjfzL/9gJ7P2cNdZRMRTVHhQ4RHJCL1orJuZf8lng0zNoa6ziIjnqPCgwiOS9vpSF3kr/5KlBaZ+vOssIuJJzvfcU+ERkU4ZSGTNW/mXruplmsa6ziIinuV8hMcL27ur8IikqRGsqSzNv7i6l2nax3UWEfE054VHIzwi0iG7mqrlr+ddGcs1sd1dZxERz6t1HcALIzzrXQcQkR2zl1n25Rt5V/hyTWyU6ywikhbWuQ7gvPCUFRU2APWuc4hI+xxgFi96OS/YJ8fEd3KdRUTSxlrXAZwXnqRVrgOIyPYdZj7++Nm8Xw/2GzvEdRYRSSsa4Ula7TqAiGzbMb53338s7/e7+AwDXGcRkbSjwpOkER4RDzvJN2v+vbl/3tMY+rrOIiJpSYUnSSM8Ih71Y//rc27OvXmsMfRynUVE0tYa1wG8cFk6aIRHxJPO9b9QEsx59DBjPPO9QkTSTxwVno00wiPiMVfmPD7zQv9zk4zxzEiwiKSnNYQicdchVHhEZAu/y7m3+Gc5/5viOoeIZARPzOJ4pfB44oshIvCv3JuLv+2fpbIjIqmy0nUA8E7h0QiPiAc8kFs0fYr/g6Nd5xCRjOKJQQ2vFB5PfDFEspe1T+WFZk7wLTradRIRyTgrXAcA7xSeKtcBRLKVj3jspbxps/bxLZvsOouIZKQlrgOAR/bhKSsqXIcOERXpdjlEW97Iu2LuPr5lR7rOIiIZq8x1APBI4Un60nUAkWyST3PjW/mXvBfwVR3uOouIZLQy1wFAhUckK/WmobY0/6JPhpt1h7jOIiIZr8x1APDOGh6AL1wHEMkG/aiNvJV/ybJ+pmG86ywikvHWE4qsdR0CvFV4NMIj0sUGEVk9M//SNb1M0/6us4hIVvDEgmXQlJZI1hjBmsrS/IsjvUzT3q6ziEjWKHMdYAMVHpEsMMpULp+Rf2lzvmnZ3XUWEckqZa4DbOClwlPmOoBIJtrbLP3y9bwr/bkmtqvrLCKSdcpcB9jAM4WnrKiwHm1AKJJS48znC1/Om9Ynx8RHuM4iIlmpzHWADTxTeJI0rSWSIkf4Pvzo6bzrh/mMHeI6i4hkrTLXATZQ4RHJQMf65r/3n9w/jPIZ+rvOIiJZzTNbznit8HjmCyOSrr7je+vtu3P/urcx9HGdRUSyWgWhSLXrEBt4rfB85DqASDo7zf/a7L/n3nqAMfR0nUVEst4C1wFa89LGgwAfuA4gkq7O8z9fck3OYxONwe86i4gI8KHrAK15rfAsBJqAfNdBRNLJNTmPzjjP/8JRxmBcZxERSfLUCI+nprTKigqjwCeuc4ikkz/k3F18fs4Lk1V2RMRjPDXC46nCk6RpLZF2uiX3n8U/yXljiuscIiKbiQMfuw7RmtemtECFR6RdHsr9Y/FR/gUqOyLiRZ8TijS4DtGaCo9I2rH26bzfzDzI97nKjoh4laems8CbhcdTi5xEvMRHPPZyXnD23r7lk11nERHZBs/9LPfcGp6yosJKYKXrHCJek0u0eXre5fP29i2f5DqLiMh2eG6Ex3OFJ8lzzVDEpR40NbyVf/EHu/pWTnSdRUSkHTz3c9yrhUfreESSetNQMyv/ooXDTPXBrrOIiLRDA/C56xCb8+IaHoD3XQcQ8YICaqvfyr+kvK9pGOc6i4hIO71NKBJzHWJzXi08c1wHEHFtMNWrZuZfuq6nad7PdRYRkR1Q6jpAW7w6pbUQWOM6hIgrO7F6RUn+xbU9TfNerrOIiOygWa4DtMWThaesqNDi0S+YSFcbbSqWFudfFs030dGus4iIdIBGeHZQiesAIt1tH7P0i//lXZ2ba2K7uM4iItIBiwlFVrkO0RYVHhGPGG8WLXwpb1o/v4mPcJ1FRKSDPDs74+XCMw9odh1CpDtM8n344dN5vxnuM3aw6ywiIp3gyeks8HDhKSsqbATecZ1DpKsd75v37sO5fwgYQ4HrLCIinaQRng7StJZktO/6Zs67I/fvY4yhj+ssIiKdVIsHd1jeQIVHxJHT/a/M+lvubeOMoYfrLCIiKTDXixsObuDVjQc38OxcoEhnTPU/89YVOU8ebgx+11lERFLEs9NZ4PERnrKiwipgsescIql0bc7DM67IeXKSyo6IZJi3XAfYFk8XniRPfwFFdsRNOXdMPyfnpcnGYFxnERFJoSZghusQ25IOhed11wFEUuH23L8Vn5pTfLTrHCIiXeAtQpF61yG2xetreABeASzoN2JJX//JvbF4kv+jKa5ziIh0kVddB9gez4/wlBUVrgTedZ1DpCMM8fhzeb+aqbIjIhnuFdcBtsfzhSfp/1wHENlRfmLRV/KumXWg74ujXGcREelClcAHrkNsjwqPSBfIJdo8Pe+y+Xv5yie5ziIi0sVeIxSxrkNsT7oUnllAxHUIkfboQVNDSf7FH+ziW32Y6ywiIt3A89NZkCaFp6yoMAr8z3UOke3pTUPNrPyLPhtqqg92nUVEpBtY4DXXIdojLQpPkqa1xNP6U7NuTv6FywaY2gNdZxER6SbvEYqsdB2iPVR4RFJgKOtWzcq/aHUf07iv6ywiIt3I85ejb5A2haesqHA58JHrHCKbG2lWVbyVf0ltT9O8p+ssIiLdLC3W70AaFZ4kjfKIp+xmKpZMz7ssnmeio11nERHpZhGgxHWI9kq3wvOS6wAiG+xryha/lndVjxwTH+k6i4iIA88TijS7DtFe6VZ4ZgCrXYcQOdgs/OTFvOv6+40d5jqLiIgjT7kOsCPSqvAkL09/xnUOyW5H+T5Y8GTeb3fyGTvIdRYREUdqSKP1O5BmhSfpSdcBJHt90zfnnQdzi3YzhgLXWUREHAoTijS6DrEj0rHwvIGmtcSBU/zFc2/L/ee+xtDbdRYREcf+6zrAjkq7wlNWVBgDnnadQ7LLmf6XZ/05547xxtDDdRYREcfqSMOLiNKu8CRpWku6zcX+p9+6PuehQ40h13UWEREPeJlQpMF1iB2V4zpAB70JrAKGuA4ime3XOQ8V/8L/8mRjMK6ziIh4RFpdnbVBWo7wJKe1dLWWdKm/5N4+/aycl6eo7IiIbNQIvOg6REekZeFJesJ1AMlcd+X+tfgU/4yjXecQEfGYVwhFal2H6Ih0ndICmI6mtSTlrH009/czDvd/MsV1EhERD0rbNbRpO8Kjq7Uk1Qzx+At5172lsiMi0qYa0ng5SdoWnqRHXAeQzOAnFv1f3lWzx/rKjnKdRUTEox4nFKl3HaKj0rrwlBUVzgAWuc4h6S2PlqYZ+Ze+s7tvxRGus4iIeNh9rgN0RloXnqR7XQeQ9NWTpvrS/Is+2tmsOdR1FhERD/uUUKTUdYjOyITCcz8QdR1C0k8f6tfPyp/6+WCz/iDXWUREPC6tR3cgAwpPWVFhJRB2nUPSywDWr52TP7W8v6k7wHUWERGPiwIPug7RWWlfeJLucR1A0scw1q6clX/Rmt6mcYzrLCIiaeD/CEUqXYforEwpPC8BFa5DiPftYlaWz8y/tL6HadnTdRYRkTSR9tNZkCGFJ7knz/2uc4i37W7Kl7yZdzl5JhpwnUVEJE2sAl5wHSIVMqLwJN0LWNchxJv2M19+/mre1T1zTHxn11lERNLIw4QiLa5DpELGFJ6yosLFJI6bENnEIebTT17I+9VAv7FDXWcREUkzGbP1S8YUnqS7XQcQb5nse/+DJ/Ju2Nln7EDXWURE0kwJociHrkOkSqYVnqeBta5DiDec6JvzzgO5N+1hDP1cZxERSUP/cB0glTKq8JQVFTYCd7nOIe790P/m3Fty/7mfMfRynUVEJA0tIY0PCm1LRhWepJuBjFhgJR1zlv+l0pty7jrIGPJdZxERSVO3EIrEXIdIpYwrPGVFheXA465ziBuX5Tw581c5D080hhzXWURE0lQdGThbknGFJ+lvrgNI9/ttzv3FF/ufOdKYjP13LSLSHR4gFKl2HSLVMvIHQ1lR4bvoEvWs8vfcW4rPyHl1ijEY11lERNKYBf7pOkRXyMjCk6RRnixxT+6fp3/XXzLFdQ4RkQzwMqHIZ65DdIVMLjwvAgtdh5CuZO3jeTcUH+t/92jXSUREMsQ/XAfoKhlbeMqKCi0Z/B8u2xni8XDetSWH+T7VyI6ISGp8RCjymusQXSVjC0/SA8Aa1yEktXKItryed9Xs/XxLjnSdRUQkg2Tk2p0NMrrwlBUVNgC3u84hqZNHS1Nx/mXv7uZbcYTrLCIiGWQF8JDrEF0powtP0r+BZtchpPN60VhXmn/RRzubNYe6ziIikmH+RCjS6DpEV8r4wlNWVFgJ3OM6h3ROX+ois/KnfjHYrD/IdRYRr4vFLePvqOWkR+oBeP2LKAfdUcu422s58t46Pl8b3+IxzTHLmc81MPa2Wg68vZbpZVEAmqKWbz5cx/631nLrvK9+dzznhQbeWZFRG/Fms0rgDtchulrGF56kPwBNrkNIxwwksmZ2/tSKAlM/1nUWkXTwzznNjBn81bf388ON/Od7PXnvvD78ZGwuv5+x5bfDu+YnTuRZcH4fXvtZL654tZG4tbyyOMqRu+bwwfm9eeiDxH3er4wRi8NBI/zd8wlJV/sLoUiD6xBdLSsKT1lR4XLgbtc5ZMcNZ21Vaf7F1b1N0xjXWUTSwfL1ccKLovzyoLyNtxkD65ssAJFGy059t9yf8+NVMb4WSBSYob199O9heLsiTq4P6lssLTGwiafg12828buv6ai6DLESuM11iO6QFYUn6Q9ARs9PZppdTdXymfmXNPYwLbu7ziKSLi79v0b+dFwPfK06zd3f6sGJjzQw8m81PPRBC8EjtywrBw738/xnUaJxy5fr4syviLEsEufru+dQVh1n4j11XHxYHs8vbOGgET526ptNPz4y2l8IRepdh+gOWfMvtqyosAK403UOaZ+9zLIv38i7wpdrYqNcZxFJFy9+1sLQ3oYJO2061fT32c289JOeLL+8L2eOy+XyV7b83e8X43MZ2dfHwXfWcekrjRyxSw5+H+T4DI98vxfvntuHH+ybwz9mN3PF4flc/kojpzxRz/MLW7rr05PUWwXc6jpEd8m2E6WLgLOBnq6DyNYdYBYveibv+v5+Y4e4ziKSTkqWxnh+YZSXFtXQGE1MYxU+Us+nq2McNjLx7f7U/XP55sNb/kKf4zP8/Zs9Nv79iHvq2GvQpr8T3zqvmdMPzGX28hgF+YbHT+nJ1x6s59t753btJyZd5a+EInWuQ3SXrBnhASgrKlyB9uXxtMPMxx8/m/frwSo7Ijvuj8f1YPnlfSm7tC+PndKTr43O4bkf9STSCJ+tSVxR9driKGOGbPmtv77FUtdsN94nxwf7DvlqpGhdg+XFRVFOPzCX+haLzyTWBjW02O755CTV1gC3uA7RnbJthAfgJuBcoJfrILKpY3zvvn9v7p93M4a+rrOIZIocn+Gub/Xg+0804DMwoIfh3pMTg9zPL2zh7YoYNxzTg5V1lm88XI/PwM59DQ99d9OB8BuKm7juqHx8xvCNPXK4ZV49Y29r4bwJeW29rHjf3whFal2H6E7G2uxr54Fg+M/Ala5zyFdO8s2af3PuzWOMUREVEelia4EAoUiN6yDdKaumtFr5E5A185Ze9yP/G3Nuzr15f5UdEZFu8cdsKzuQpYWnrKhwFYkjJ8Sxc/wvlvwx5+4JxqBNPUREut6XwM2uQ7iQlYUnqQidpO7UlTmPz5yW88jhxmTlWjIRERemEYpk5ckDWVt4yooKq4GQ4xhZ64ac+4qn5jx3lDHZ+29QRKSbzSYUedx1CFey/YfN7cAnrkNkm3/m3lx8es5rU1znEBHJMll9sU5WF56yosIocIXrHNnkvtybpp/sn6WyIyLSvZ4iFClxHcKlrC48AGVFhS8Dr7jOkfms/W9eaMYx/vePdp1ERCTLtADXuA7hWtYXnqTLgZjrEJnKRzz2ct60koN9n012nUVEJAvdSiiy2HUI11R4gLKiwo+BO1znyEQ5RFveyLti7hjf0iNdZxERyULrgBtch/ACFZ6v/Aaodh0ik+TT3PhW/iXvBXxVh7vOIiKSpW4kFFnrOoQXqPAklRUVrgZ+5zpHpuhNQ21p/kWfDDfrDnGdRUQkSy0mSzcZbIsKz6b+DXzuOkS660dtZFb+1LJBpma86ywiIlnsAkKRZtchvEKFp5WyosJmdJl6pwwisnp2/kWV/UzD/q6ziIhksccJRV51HcJLVHg2U1ZU+DzwrOsc6WgEaypL8y+O9DJNe7vOIiKSxSLApa5DeI0KT9umAutdh0gno0zl8hn5lzbnm5bdXWcREcly1xKKVLoO4TUqPG0oKyosB6a5zpEu9jZLv3w970p/ront6jqLiEiWm0vi2CTZjArP1t0GlLoO4XUHms8/ezlvWp8cEx/hOouISJaLAecSisRdB/EiFZ6tKCsqtMA5JLbkljZM9H300TN51w/1GTvEdRYREeFfhCLvuQ7hVSo821BWVPgRcJPrHF50rG/+e4/m3rirz9DfdRYREWEZcL3rEF6mwrN9vwc+cx3CS072lbx9d+5f9zaGvq6ziIgIABcTitS6DuFlKjzbUVZU2ERiasu6zuIFp/lfm/2P3FsOMIaerrOIiAgAzxOKPOs6hNep8LRDWVFhMXCv6xyuned/vuR3OfcdYgx5rrOIiAgAa4HzXYdIByo87XcVUOU6hCvX5Dw645qcx44wBr/rLCIistEFhCIVrkOkAxWediorKlwHnOs6hwt/yLm7+PycFyYbg3GdRURENnqMUORx1yHShQrPDigrKnwOuNt1ju50S+4/i3+S88YU1zlERGQTFcAFrkOkExWeHXcpWXKi+oO5fywu9M9R2RER8Z5fEIqscx0inajw7KCyosI64KdA1HWWrmPtU3m/mTHZv0BlR0TEe24jFHnFdYh0o8LTAWVFhXOB37nO0RV8xGP/lxcsmeBbNNl1FhER2cIi4ErXIdKRCk/H3QjMch0ilXKJNr+Zd/m8fXzLjnSdRUREthADTicUqXcdJB2p8HRQWVFhDDgNyIidLXvQ1DAz/5IPRvlWTnSdRURE2nQTochs1yHSlQpPJ5QVFX4BXOw6R2f1pqG2NP/ihcPNuoNdZxERkTa9C4Rch0hnxlqdmNBZgWD4v8D3XefoiAJqq9/Kv6S8r2nYz3UWERFpUw1wMKGIznXsBI3wpMa5JPZESCuDqV41K39qlcqOiIinnaWy03kqPClQVlS4hsSl6jHXWdprJ1avKMm/uLaXad7bdRYREdmqmwlFnnQdIhOo8KRIWVHhdGCa6xztMdpULC3Ovyyab6KjXWcREZGtmosuQU8ZreFJsUAw/BTwPdc5tmYfs/SLcN61Pf0mPsJ1FhER2aq1wHhCkaWug2QKjfCk3s+Bha5DtGW8WbTwpbxp/VR2REQ8zQI/U9lJLRWeFCsrKqwhMcLjqf15Jvk+/PDpvN8M9xk72HUWERHZpiJCkZdch8g0KjxdoKyo8GPgLNc5Nvi67+33Hs79Q8AYClxnERGRbZoO/Np1iEykwtNFyooKnwD+4TrHd3xvvX1n7t/2NoY+rrOIiMg2VQI/JhRJmyt+04kKT9e6Cpjp6sV/5n919t9zbz3AGHq6yiAiIu0SJVF2Kl0HyVQqPF2orKgwCvwQWNHdr32h/9m3bsi5/xBjyOvu1xYRkR02lVBkuusQmUyFp4uVFRVWAj8AWrrrNafl/GfGlTlPTDIGf3e9poiIdNg/CEXucB0i06nwdIOyosIS4PzueK2inDunn5sTnmwMpjteT0REOuUl4ArXIbKBNh7sRoFg+A904W7Mt+X+vfgE/7wpXfX8IiKSUh8CRxCK1LgOkg00wtO9rgMe64onfjj3Dyo7IiLpYyXwLZWd7qMRnm4WCIbzgdeBSal4PkM8/kzeb0rG+RYflYrnExGRLtcEfI1QpNR1kGyiwuNAIBgeBMwG9ujM8/iJRV/OC87Zy1eekvIkIiLd4jRCkf+4DpFtNKXlQFlR4RrgRBKHw3VILtHm6XmXzVfZERFJKzeq7LihwuNIWVHhIuA7JIY2d0gPmhpK8i/+YBff6sNSHkxERLrKf9GxEc5oSsuxQDD8E+BhaN9l5L1pqHkr/5IvBpjaA7s2mYiIpNB04JuEIjv8S66khkZ4HCsrKnwEuL499y2gtnpO/oXLVHZERNLKfODbKjtuaYTHIwLB8J3A2Vv7+GCqV83Mv3RdT9O8VzfGEhGRzlkIHEUossp1kGynER7vOA94oq0P7MyqFaX5F9eq7IiIpJVlwNdVdrxBIzweEgiGc4HngBM23LabqVjySt41/lwTG+kumYiI7KBVJEZ2FroOIgka4fGQsqLCFuD7wEyAMWbJ4tfyrspX2RERSSs1wAkqO96iwuMxZUWFDcBJE30fPRfOu7a/39jhrjOJiEi7NZJYoDzfdRDZlKa0vCpUMJjEZYz7OU4iIiLtEwO+TyjynOsgsiWN8HhVKLIaOJbECn8REfE2C5ylsuNdKjxeFopUkSg9X7iOIiIiWxUHfkko8oDrILJ1mtJKB6GCUcAMYFfXUUREZBNx4ExCkQddB5Ft0whPOghFlgBfA5a6jiIiIhvFSJx8rrKTBlR40kUoshg4CljkOoqIiBAFfkwo8qjrINI+KjzpJBRZCkwGPnIdRUQki7UAPyQUedJ1EGk/FZ50E4pUAlNIHEYnIiLdq5nEpefPuA4iO0aFJx2FImtIrOkpcR1FRCSLNALfIRR5wXUQ2XEqPOkqFFkPfAN43XUUEZEs0EBiB+WXXQeRjlHhSWehSB1QCLzoOoqISAarAQoJRV5zHUQ6ToUn3YUiTcD3gCdcRxERyUBVwBRCkTddB5HOUeHJBKFIC/Bj4F7XUUREMshi4AhCkXddB5HO007LmSZU8FvgetcxRETS3HzgREKRla6DSGqo8GSiUMEZwF1ArusoIiJp6DUSl57XuA4iqaMprUyUOMDum0C14yQiIunmfhILlFV2MoxGeDJZqGAM8BIQcJxERCQdhAhFfus6hHQNFZ5MFyoYBrwAHOI6ioiIR7UAZydHxyVDaUor04UiVcDRwLNug4iIeFI1iSkslZ0Mp8KTDUKReuD7wD8cJxER8ZKPgUO0oWB20JRWtgkVTCVRfPyOk4iIuPQscLoWJ2cPjfBkm1Dk38DXAe0tISLZyAIh4HsqO9lFIzzZKlQwEvgvcJjrKCIi3aQG+BmhyHOug0j30whPtgpFlgOTgdtdRxER6QaLgMNUdrKXRngEQgU/B24DejhOIiLSFV4CfkIoEnEdRNzRCI9AKHI/MAkocxtERCTl/gh8S2VHNMIjXwkVDAQeAb7hOoqISCdVA2cRijztOoh4g0Z45CuhyFrgROD3JK5kEBFJRyXAgSo70ppGeKRtoYJC4D5giOsoIiLtFAf+QOJMrJjrMOItKjyydYlzuO4DTnAdRURkOyqA0whF3nQdpDVjTAxYAOQAnwBnWGvrt3LfccBO1tqXdvA1QkCttfYvnUub2TSlJVsXilQRipwIXAQ0uo4jIrIVLwAHeK3sJDVYa8dZa/cHmoHztnHfcSSWFWzBGJPTBdmyigqPbF9id+aDgfddRxERaaUJuIRQ5NuEImtch2mHmcAexpjexph7jTFzjTHvGmNONsbkATcApxpj3jPGnGqMCRljHjLGlAAPGWMCxpg3jDEfGGNeN8bs6vbTSS8qPNI+ochHJHZl/jta0Cwi7i0EJhKK/Mt1kPZIjtCcQGJ66zrgDWvtocAxwJ+BXOB64PHkiNDjyYfuCxxnrf0xcDPwgLX2AOA/QFp87l6hwiPtF4o0EYpcTuKy9QrXcUQka90FTCAUec91kHboaYx5D3gbWArcAxwPBJO3Tyex6evWRmuet9Y2JN8/nMTWIQAPAUd2TeTMpDlB2XGhyGuECg4g8U3nu67jiEjWWAL8klDkf66D7IAGa+241jcYYwzwfWvtws1ub+tsw7ouzJZVNMIjHROKrCEU+R5wJrDOdRwRyWgWuAXYP83Kzta8AlyULD4YY8Ynb68B+m7jcaXAj5Lv/5TEmiBpJxUe6ZzEsRRjgCcdJxGRzPQ5cDShyFRCkVrXYVLkdyTW7HxgjPko+XeAN4F9NyxabuNxFwFnGmM+AH4GXNItaTOE9uGR1AkVfJvEb2EjXUcRkbQXB/4B/IpQpGE79xXZLhUeSa1QQV+gCDgfMI7TiEh6+gT4BaHIbNdBJHOo8EjXCBVMIrGoeYzrKCKSNqIkLtH+LaFIk+swkllUeKTrhArygGuBaUCe4zQi4m2lwFRCkXddB5HMpMIjXS9UsC+J0Z4jXEcREc+pBK4BHiIU0Q8k6TIqPNI9QgUGOI3E+p6dHKcREfdaSOwUfAOhyHrXYSTzqfBI9woV9AaCwJUkdhcVkezzP+BiQpFPXAeR7KHCI26ECkYBfwJ+6DqKiHSbJcAVhCJPuQ4i2UeFR9wKFRxFYq+NgxwnEZGu00jiF5wi7akjrqjwiHuhAh/wc+APwDC3YUQkhSzwBDCNUORL12Eku6nwiHckNi28DrgUyHcbRkQ66WXg2jQ50VyygAqPeE+oYBfgVyQOJs11nEZEdkwJiREdHWwpnqLCI94VKhgNXE/ikDy/4zQism3vA9cRioRdBxFpiwqPeF+oYC/gN8CPAJ/jNCKyqcUkfjF5VBsHipep8Ej6CBXsB/wW+B46mFTEtRXADcA9hCItrsOIbI8Kj6SfUME4Et9ov+U4iUg2KgP+AtyrS8wlnajwSPoKFRxK4nDSb6MRH5GutgC4CXicUCTqOozIjlLhkfSXWONzBXA6Oq5CJNVmktgw8CXXQUQ6Q4VHMkeoYAgwFbgAGOw4jUg6s8CLJIpOqeswIqmgwiOZJ1TQk8TOzZcBe7oNI5JWosCjwE2EIh+5DiOSSio8krkSR1acTOJk9iMcpxHxspXAPcDthCJLXYcR6QoqPJIdQgWHAxeRuKRdx1aIJMwEbgWeJhRpdh1GpCup8Eh2CRUMIrG4+WxgjOM0Ii7UAA8BtxGKfOg6jEh3UeGR7BUqOBI4B/gBurpLMt/7wG3AfwhFal2HEeluKjwioYIBwGkkys/+jtOIpFID8BSJ0RxdbSVZTYVHpLXEWp+zgVOBXo7TiHREHHgD+A/wFKFIjeM8Ip6gwiPSllBBbxJHV/wI+CZa6Cze9w6JkvMoocgK12FEvEaFR2R7QgUFwHdJlJ9jgRy3gUQ2+hJ4BHiYUORT12FEvEyFR2RHhAoGA98nUX4mAz63gSQLrQaeJFFytC5HpJ1UeEQ6KlQwgsQVXqcCE1H5ka6zEHgeeAEoJRSJOc4jknZUeERSIXGO1wnAicDxwAC3gSTNxYASEgXneUKRzxznEUl7KjwiqRYq8JM4yuLE5NsBbgNJmqgBXiExkhMmFFnrOI9IRlHhEelqoYKRfFV+jgX6uA0kHhEH3gPeBF4Fput4B5Guo8Ij0p1CBXnAJBILnieTWPuj/X6yQxz4gETBmQ7MIBSpdhlIJJuo8Ii4FCrIBQ4CjiJRgI5E638yhQUWsGnB0TSViCMqPCJeEiowwH4kCtCGt5FOM0l7VQPzgbeBuUAxocgap4lEZCMVHhGvCxXsAowDxif/HAeMdhdIgAiJnY03FJy3CUUWu40kItuiwiOSjhK7P49j0yK0L5DrLFPmqgI+IVFw3iZRchYRiuibp0gaUeERyRSJBdH7Jd/2aPW2OzDYYbJ00AQsIrHB36ZvWlgskhFUeESyQWJEaEP52WOz90cAxl24bhEHVgHlybflwGfApySKzRJCkbi7eCLS1VR4RLJdYmRo2Dbehrd6f6CjlFvTAKwH1gIVJMpMW3+uIBSJugopIu6p8IhI+yUuox8C9AP6tnrrk/yzF9AD6Nnqz54kRpDiJC7Vjm/21tZtTSSKzOZvkU3+rhIjIu2kwiMiIiIZT6c7i4iISMZT4REREZGMp8IjIiIiGU+FR0RERDKeCo+IiIhkPBUeERERyXgqPCIiIpLxVHhEREQk46nwiIiISMZT4REREZGMp8IjIiIiGU+FR0RERDKeCo+IiIhkPBUeERERyXgqPCIiIpLxVHhEREQk46nwiIiISMZT4REREZGMp8IjIiIiGU+FR0RERDKeCo+IiIhkPBUeERERyXgqPCIiIpLxVHhEREQk4/0/WUYfYUDmuC0AAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 720x720 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Get the counts of each category in the column\n",
        "counts = df['fuel'].value_counts()\n",
        "\n",
        "# Create a pie chart\n",
        "fig, ax = plt.subplots(figsize=(10, 10))\n",
        "plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%' , startangle=45 , textprops={'fontsize': 10})\n",
        "plt.axis('equal')\n",
        "plt.title('Car_type')\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nh7UDfAM2yOM",
        "outputId": "97924125-b0a9-42d0-8093-d1ccf307763f"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "The number of unique car count variables is: 1491\n"
          ]
        }
      ],
      "source": [
        "unique_car = df[\"name\"].nunique()\n",
        "print(\"The number of unique car count variables is:\", unique_car)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0XxcI08LLwZi",
        "outputId": "fee2387f-7573-4487-b0a8-5559c60037ff"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-cd8fd47e-ff55-4ad4-97bd-2c90901aa6b8\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>owner</th>\n",
              "      <th>count</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>First Owner</td>\n",
              "      <td>2832</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Fourth &amp; Above Owner</td>\n",
              "      <td>81</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Second Owner</td>\n",
              "      <td>1106</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Test Drive Car</td>\n",
              "      <td>17</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Third Owner</td>\n",
              "      <td>304</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-cd8fd47e-ff55-4ad4-97bd-2c90901aa6b8')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-cd8fd47e-ff55-4ad4-97bd-2c90901aa6b8 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-cd8fd47e-ff55-4ad4-97bd-2c90901aa6b8');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "                  owner  count\n",
              "0           First Owner   2832\n",
              "1  Fourth & Above Owner     81\n",
              "2          Second Owner   1106\n",
              "3        Test Drive Car     17\n",
              "4           Third Owner    304"
            ]
          },
          "execution_count": 16,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "car_owner = df.groupby(\"owner\")[\"name\"].count().reset_index().rename(columns={\"name\": \"count\"})\n",
        "car_owner\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QtZAdq3UJehr",
        "outputId": "d6fc062b-3c5a-4fe7-ab02-3a57da554ddf"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<Axes: title={'center': 'car_owner Type'}, xlabel='owner', ylabel='count'>"
            ]
          },
          "execution_count": 17,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 936x576 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "plt.figure(figsize=(13,8))\n",
        "plt.title('car_owner Type')\n",
        "sns.barplot(x='owner',y='count',data=car_owner)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5J7b0GyVKdW9",
        "outputId": "c4393d7b-49d0-4b1e-991b-bcaee2231cb4"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-187fffc5-24d0-45dd-b7db-568ef8ac44f3\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>name</th>\n",
              "      <th>year</th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Maruti 800 AC</td>\n",
              "      <td>2007</td>\n",
              "      <td>60000</td>\n",
              "      <td>70000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Maruti Wagon R LXI Minor</td>\n",
              "      <td>2007</td>\n",
              "      <td>135000</td>\n",
              "      <td>50000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-187fffc5-24d0-45dd-b7db-568ef8ac44f3')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-187fffc5-24d0-45dd-b7db-568ef8ac44f3 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-187fffc5-24d0-45dd-b7db-568ef8ac44f3');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "                       name  year  selling_price  km_driven    fuel  \\\n",
              "0             Maruti 800 AC  2007          60000      70000  Petrol   \n",
              "1  Maruti Wagon R LXI Minor  2007         135000      50000  Petrol   \n",
              "\n",
              "  seller_type transmission        owner  \n",
              "0  Individual       Manual  First Owner  \n",
              "1  Individual       Manual  First Owner  "
            ]
          },
          "execution_count": 18,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df.head(2)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "E4bMejwWPD32",
        "outputId": "c25be519-4d97-4312-ab35-8334b6937f9c"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-f9784fb2-9aa0-4393-85df-6131f7648967\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>fuel</th>\n",
              "      <th>selling_price</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>CNG</td>\n",
              "      <td>11086997</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Diesel</td>\n",
              "      <td>1440559925</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Electric</td>\n",
              "      <td>310000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>LPG</td>\n",
              "      <td>3859999</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Petrol</td>\n",
              "      <td>732095612</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-f9784fb2-9aa0-4393-85df-6131f7648967')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-f9784fb2-9aa0-4393-85df-6131f7648967 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-f9784fb2-9aa0-4393-85df-6131f7648967');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "       fuel  selling_price\n",
              "0       CNG       11086997\n",
              "1    Diesel     1440559925\n",
              "2  Electric         310000\n",
              "3       LPG        3859999\n",
              "4    Petrol      732095612"
            ]
          },
          "execution_count": 19,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "selling_price = df.groupby(\"fuel\")[\"selling_price\"].sum().reset_index()\n",
        "selling_price"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zPCYDeYLQFjH",
        "outputId": "4cfdd6e0-6957-41d3-ac0e-c6eee6b170ae"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<Axes: title={'center': 'max_sell_CAR_Type'}, xlabel='fuel', ylabel='selling_price'>"
            ]
          },
          "execution_count": 20,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 936x576 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "plt.figure(figsize=(13,8))\n",
        "plt.title('max_sell_CAR_Type')\n",
        "sns.barplot(x='fuel',y='selling_price',data=selling_price)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SXPuqXY3QzJE",
        "outputId": "8271b647-5117-4edc-8a95-ffba1131c8f8"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "owner        selling_price\n",
            "First Owner  300000           114\n",
            "             550000            87\n",
            "             600000            84\n",
            "Name: selling_price, dtype: int64\n",
            "owner         selling_price\n",
            "Second Owner  150000           43\n",
            "              300000           42\n",
            "              450000           41\n",
            "Name: selling_price, dtype: int64\n",
            "owner                 selling_price\n",
            "Fourth & Above Owner  110000           7\n",
            "                      70000            6\n",
            "                      250000           6\n",
            "Name: selling_price, dtype: int64\n",
            "owner        selling_price\n",
            "Third Owner  150000           14\n",
            "             120000           11\n",
            "             110000            9\n",
            "Name: selling_price, dtype: int64\n",
            "owner           selling_price\n",
            "Test Drive Car  541000           1\n",
            "                635000           1\n",
            "                700000           1\n",
            "Name: selling_price, dtype: int64\n"
          ]
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 360x216 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 360x216 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 360x216 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 360x216 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 360x216 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Chart - 4 visualization code\n",
        "newlist=df['owner'].unique()\n",
        "y=list(newlist)\n",
        "# Applying for loop operation\n",
        "for x in y:\n",
        "  sub_1=df.loc[df['owner'] == x]\n",
        "  p_sub1=sub_1.groupby(['owner'])['selling_price'].value_counts().head(3)\n",
        "  print(p_sub1)\n",
        "  plt.figure(figsize=(5,3))\n",
        "  p_sub1.plot(kind='bar')\n",
        "  plt.title('Top 3 sold car of '+str(x))\n",
        "  plt.xlabel(\"car\")\n",
        "  plt.ylabel(\"Counting\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1qs6mWLOWup-",
        "outputId": "66f62537-7569-4e5b-832e-0df496cd27a6"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-ceaa2494-7e55-4129-b3d3-ebb3226584d4\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>seller_type</th>\n",
              "      <th>count</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Dealer</td>\n",
              "      <td>994</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Individual</td>\n",
              "      <td>3244</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Trustmark Dealer</td>\n",
              "      <td>102</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-ceaa2494-7e55-4129-b3d3-ebb3226584d4')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-ceaa2494-7e55-4129-b3d3-ebb3226584d4 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-ceaa2494-7e55-4129-b3d3-ebb3226584d4');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "        seller_type  count\n",
              "0            Dealer    994\n",
              "1        Individual   3244\n",
              "2  Trustmark Dealer    102"
            ]
          },
          "execution_count": 22,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df_plot = df.groupby(['seller_type'])['name'].count().reset_index().rename(columns={\"name\": \"count\"})\n",
        "\n",
        "df_plot"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Nx0kEkUkGD7I",
        "outputId": "0fea04db-e164-4b45-b0e7-5d17a16bfa70"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<Axes: xlabel='km_driven', ylabel='selling_price'>"
            ]
          },
          "execution_count": 23,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 720x432 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "#visualization code\n",
        "#Writing a code for plotting line plot between the target variable and age, cigsPerDay, and heartRate\n",
        "plt.figure(figsize=(10, 6))\n",
        "sns.lineplot(x='km_driven', y=\"selling_price\", data=df.head(200))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "odI2qtb2MA2J",
        "outputId": "523ced8d-2792-4015-9a23-a886d74a03d2"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-1ce46578-a7d4-476c-9cfe-ae92e0abfb3c\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>name</th>\n",
              "      <th>year</th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Maruti 800 AC</td>\n",
              "      <td>2007</td>\n",
              "      <td>60000</td>\n",
              "      <td>70000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-1ce46578-a7d4-476c-9cfe-ae92e0abfb3c')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-1ce46578-a7d4-476c-9cfe-ae92e0abfb3c button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-1ce46578-a7d4-476c-9cfe-ae92e0abfb3c');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "            name  year  selling_price  km_driven    fuel seller_type  \\\n",
              "0  Maruti 800 AC  2007          60000      70000  Petrol  Individual   \n",
              "\n",
              "  transmission        owner  \n",
              "0       Manual  First Owner  "
            ]
          },
          "execution_count": 24,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df.head(1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ZrfM5x0PL7Yz"
      },
      "outputs": [],
      "source": [
        "#Here we can see that on an average selling price of each fuel type car in each year \n",
        "# Chart - 9 visualization code\n",
        "plt.figure(figsize=(30,10))\n",
        "ax=sns.barplot(x=df['year'], y=df['selling_price'],hue=df['fuel'])\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "KK4BdTLzPxNx"
      },
      "outputs": [],
      "source": [
        "#Lets check the selling price of manual and automatic cars\n",
        "plt.figure(figsize=(10,16))\n",
        "ax=sns.boxplot(x=df['transmission'], y=df['selling_price'])\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bqKmuiVOa-4f"
      },
      "source": [
        "## ***4. Feature Engineering & Data Pre-processing***\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5rZnK3DHtOZu"
      },
      "outputs": [],
      "source": [
        "df1 = df [['year', 'selling_price', 'km_driven',\n",
        "       'fuel', 'seller_type', 'transmission', 'owner']]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-kvPr3m9jV9C"
      },
      "outputs": [],
      "source": [
        "df1['Current_Year'] = 2022"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bi6j_l0sjrIL"
      },
      "outputs": [],
      "source": [
        "df1.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "05ckY792jwzO"
      },
      "outputs": [],
      "source": [
        "#Creating our new column no_of_year\n",
        "df1['Year_old'] = df1['Current_Year'] - df1['year']"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9DdXY_6Jkm3Y"
      },
      "outputs": [],
      "source": [
        "#One Hot Encoding for Categorical variables by creating dummy variables\n",
        "df1 = pd.get_dummies(df1, drop_first = True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "tfpBkBHEu0YC"
      },
      "outputs": [],
      "source": [
        "df1.drop(['year','Current_Year'], axis=1, inplace=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4Spln_pRu_8F"
      },
      "outputs": [],
      "source": [
        "df1.head()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nmf_CokoZDkn"
      },
      "source": [
        "### 2. Handling Outliers"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "mARcH0e0F02K"
      },
      "outputs": [],
      "source": [
        "#plotting distribution plot of target Variables\n",
        "sns.distplot(x=df1.selling_price)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "A_1znpevd7wc"
      },
      "outputs": [],
      "source": [
        "#plotting distribution plot of target Variables\n",
        "sns.distplot(x=df1.km_driven)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WCNtri6GZ8fr"
      },
      "source": [
        "A Distplot or distribution plot, depicts the variation in the data distribution. Seaborn Distplot represents the overall distribution of continuous data variables i.e. data distribution of a variable against the density distribution. In above graph we can see that our graph have right tail(right skewed) it means tha our most of data centered near peak of the graph but there is a some highly expensive price that leads to a seperate trends that's why our graph showing right trends."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "qXq5iDUXhvja"
      },
      "outputs": [],
      "source": [
        "\n",
        "# calculate the upper and lower limits for the km_driven column using the capping method\n",
        "Q1 = df1[\"km_driven\"].quantile(0.25)\n",
        "Q3 = df1[\"km_driven\"].quantile(0.75)\n",
        "IQR = Q3 - Q1\n",
        "upper_limit = Q3 + 1.5*IQR\n",
        "lower_limit = Q1 - 1.5*IQR\n",
        "\n",
        "# replace outliers above the upper limit with the nearest non-outlier value, and outliers below the lower limit with the nearest non-outlier value\n",
        "df1[\"km_driven\"] = np.where(df1[\"km_driven\"] > upper_limit, df1[\"km_driven\"].quantile(0.95), df1[\"km_driven\"])\n",
        "df1[\"km_driven\"] = np.where(df1[\"km_driven\"] < lower_limit, df1[\"km_driven\"].quantile(0.05), df1[\"km_driven\"])\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-4ACsceSregF"
      },
      "source": [
        "In above, the upper and lower limits are calculated using the interquartile range (IQR) multiplied by 1.5. Any values above the upper limit or below the lower limit are replaced with the 99th or 1st percentile value, respectively, using the NumPy where function"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "wZ91fHyjfMUW"
      },
      "outputs": [],
      "source": [
        "sns.distplot(x=df1.km_driven)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "M3SIrlJiiVS2"
      },
      "outputs": [],
      "source": [
        "df1[\"km_driven_sqrt\"] = np.sqrt(df1[\"km_driven\"])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "3eIw7dlkiY3H"
      },
      "outputs": [],
      "source": [
        "sns.distplot(x=df1.km_driven_sqrt)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5miW3JNVcwHr"
      },
      "source": [
        "After transforming the data now we can see that our data looks good and Logarithmic transformation converted our data to a normal distribution. It is a type of data transformation that can be applied to data that has a wide range of values or is skewed in one direction."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "rDMRcUBUWbV3"
      },
      "outputs": [],
      "source": [
        "# take the natural logarithm of a column called \"column_to_transform\"\n",
        "df1[\"selling_price\"] = np.log(df1[\"selling_price\"])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ajkgsTHRWm8w"
      },
      "outputs": [],
      "source": [
        "#plotting distribution plot of target Variables\n",
        "sns.distplot(x=df1.selling_price)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oRtVZIdHWidz"
      },
      "source": [
        "Logarithmic transformation is a mathematical operation used in data analysis and modeling to convert data that is not normally distributed to a normal distribution. It is a type of data transformation that can be applied to data that has a wide range of values or is skewed in one direction."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "56G1gwAxrwRx"
      },
      "outputs": [],
      "source": [
        "df1.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "CETuzHHbtVLE"
      },
      "outputs": [],
      "source": [
        "#dropping un_necessary colmns\n",
        "df1.drop(['km_driven'], axis=1, inplace=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "NoD-cf8PUflp"
      },
      "outputs": [],
      "source": [
        "df1.columns"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0-5D5IiPVWUA"
      },
      "source": [
        "#### 2. Feature Selection"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "CmoHQ4gOMuxN"
      },
      "outputs": [],
      "source": [
        "\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "\n",
        "# define your feature matrix X and target variable y\n",
        "X = df1.drop('selling_price', axis=1)\n",
        "y = df1['selling_price']\n",
        "\n",
        "# train a Random Forest regressor\n",
        "rf = RandomForestRegressor()\n",
        "rf.fit(X, y)\n",
        "\n",
        "# get feature importances and put them into a DataFrame\n",
        "importances = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})\n",
        "\n",
        "# sort the DataFrame by importance (descending)\n",
        "importances = importances.sort_values('importance', ascending=False)\n",
        "\n",
        "# print the top 10 features by importance\n",
        "print(importances.head(16))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RMUv2x4lUYpO"
      },
      "outputs": [],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# create a bar chart of feature importances\n",
        "plt.bar(importances['feature'], importances['importance'])\n",
        "plt.xticks(rotation=90)\n",
        "plt.ylabel('Importance')\n",
        "plt.xlabel('Feature')\n",
        "plt.show()\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Dqr-N7fcU2CC"
      },
      "outputs": [],
      "source": [
        "#removing un-necessary features\n",
        "df1.drop(['fuel_LPG', 'owner_Test Drive Car','fuel_Electric' ], axis=1, inplace=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "f_X_eVmwWO2B"
      },
      "outputs": [],
      "source": [
        "df1.columns"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oHADttELXx9O"
      },
      "source": [
        "### 6. Data Scaling"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xQwpYQTPh93v"
      },
      "source": [
        "StandardScaler is a popular method for scaling numerical data, typically used in machine learning and data analysis. It transforms your data so that it has a mean of 0 and a standard deviation of 1.\n",
        "\n",
        "StandardScaler is a commonly used scaling method that scales the data to have zero mean and unit variance. This ensures that the features are on the same scale and have similar ranges. Scaling is generally considered a good practice in machine learning and data analysis, and is often a necessary step in the preprocessing pipeline"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "yElGFGESX9TS"
      },
      "outputs": [],
      "source": [
        "#making copy\n",
        "df2=df1.copy()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9-5oKCYdYE6P"
      },
      "outputs": [],
      "source": [
        "#importing libraray\n",
        "from sklearn.preprocessing import StandardScaler"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "QKfm-rCbaWMy"
      },
      "outputs": [],
      "source": [
        "#convering data type\n",
        "df2['Year_old'] = df2['Year_old'].astype(float)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "MFR-mu2hYIq-"
      },
      "outputs": [],
      "source": [
        "# Scaling your data\n",
        "#applying standardScaler \n",
        "scaler = StandardScaler()\n",
        "df3 = scaler.fit_transform(df2)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "V4okfEs_g0N7"
      },
      "outputs": [],
      "source": [
        "#converting back to dataframe\n",
        "df3 = pd.DataFrame(df3, columns=df2.columns)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9cYyXGGmgQYk"
      },
      "outputs": [],
      "source": [
        "df3.head()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aKrZRyZBioWB"
      },
      "source": [
        "### 8. Data Splitting"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "H84IlC-Xw9_W"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "from numpy import math\n",
        "\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.metrics import r2_score\n",
        "from sklearn.metrics import mean_squared_error\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "x6Rjicxri9eS"
      },
      "outputs": [],
      "source": [
        "# Split your data to train and test. Choose Splitting ratio wisely.\n",
        "x, y = df3.loc[:, df3.columns != 'selling_price'], df3['selling_price']"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cW_XUONJuAqC"
      },
      "outputs": [],
      "source": [
        "#importing library to split\n",
        "from sklearn.model_selection import train_test_split\n",
        "#dividing the data for training and testing\n",
        "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 0)\n",
        "print(x_train.shape)\n",
        "print(x_test.shape)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VfCC591jGiD4"
      },
      "source": [
        "## ***5. ML Model Implementation***"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "h1DkwNEcdmkH"
      },
      "outputs": [],
      "source": [
        "# create a dataframe to store metrics related to models\n",
        "metrics_table = pd.DataFrame(columns=['Regression_Model', 'Train_R2', 'Test_R2', 'Train_RMSE', 'Test_RMSE', 'Train_RMSPE', 'Test_RMSPE'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "kp-j2OO4d2Ez"
      },
      "outputs": [],
      "source": [
        "# define a function to calculate root mean squared percentage error\n",
        "# returns an array\n",
        "def calculate_rmspe(y, y_pred):\n",
        "  return (np.sqrt(np.mean(np.square(y.to_numpy() - y_pred))) / np.mean(y.to_numpy())) * 100"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bIhSdi45d4GT"
      },
      "outputs": [],
      "source": [
        "# define a function to calculate metrics\n",
        "# returns a dictionary\n",
        "def calculate_model_metrics(y_train, y_train_pred, y_test, y_test_pred):\n",
        "  metrics_dict = {}\n",
        "\n",
        "  metrics_dict['Train_R2'] = r2_score(y_train, y_train_pred)\n",
        "  metrics_dict['Test_R2'] = r2_score(y_test, y_test_pred)\n",
        "  metrics_dict['Train_RMSE'] = mean_squared_error(y_train, y_train_pred, squared=False)\n",
        "  metrics_dict['Test_RMSE'] = mean_squared_error(y_test, y_test_pred, squared=False)\n",
        "  metrics_dict['Train_RMSPE'] = calculate_rmspe(y_train, y_train_pred)\n",
        "  metrics_dict['Test_RMSPE'] = calculate_rmspe(y_test, y_test_pred)\n",
        "\n",
        "  return metrics_dict"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "OB4l2ZhMeS1U"
      },
      "source": [
        "### ML Model - 1 Linear Regression"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7ebyywQieS1U"
      },
      "outputs": [],
      "source": [
        "# ML Model - 1 Implementation\n",
        "# Fit the Algorithm\n",
        "# Predict on the model"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "mSt7I_6J_izA"
      },
      "outputs": [],
      "source": [
        "from sklearn.linear_model import LinearRegression\n",
        "regression_model = LinearRegression()\n",
        "regression_model.fit(x_train, y_train)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bcuFdqH6CJ1Q"
      },
      "outputs": [],
      "source": [
        "regression_model.score(x_test, y_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "T4kldYVascK3"
      },
      "outputs": [],
      "source": [
        "#checking prediction\n",
        "y_train_pred = regression_model.predict(x_train)\n",
        "y_test_pred = regression_model.predict(x_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "eaeQJeAgeL-W"
      },
      "outputs": [],
      "source": [
        "#calculating for model metrics\n",
        "model_evaluation = calculate_model_metrics(y_train, y_train_pred, y_test, y_test_pred)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "vScWpS32ePdF"
      },
      "outputs": [],
      "source": [
        "#storing data in metric tables\n",
        "metrics_table.loc[len(metrics_table.index)] = ['Linear', model_evaluation['Train_R2'], model_evaluation['Test_R2'], \n",
        "                                                         model_evaluation['Train_RMSE'], model_evaluation['Test_RMSE'], \n",
        "                                                         model_evaluation['Train_RMSPE'], model_evaluation['Test_RMSPE']]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ArJBuiUVfxKd"
      },
      "source": [
        "#### 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iGrWcAk3CRIK"
      },
      "source": [
        "We have used Linear Regression as our first model, Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "wQ6v-f0sKMUu"
      },
      "outputs": [],
      "source": [
        "metrics_table"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4qY1EAkEfxKe"
      },
      "source": [
        "#### 2. Cross- Validation & Hyperparameter Tuning"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Dy61ujd6fxKe"
      },
      "outputs": [],
      "source": [
        "# ML Model - 1 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)\n",
        "# Fit the Algorithm\n",
        "# Predict on the model\n",
        "\n",
        "# cross validation using k fold technique\n",
        "from sklearn.model_selection import cross_val_score\n",
        "for i in [3,5,10]:\n",
        "  score = cross_val_score(LinearRegression(), x_train, y_train,cv=i)\n",
        "  print(np.average(score))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Uf3AQd_5vqef"
      },
      "outputs": [],
      "source": [
        "#hyper parameter tuning using RandomizedSearchCV\n",
        "from sklearn.model_selection import RandomizedSearchCV\n",
        "from scipy.stats import randint\n",
        "\n",
        "lr = LinearRegression()\n",
        "\n",
        "param_distributions = {\"fit_intercept\": [True, False],\n",
        "                        \"copy_X\": [True, False],\n",
        "                       \"positive\": [True, False]}\n",
        "search = RandomizedSearchCV(lr, param_distributions).fit(x, y)\n",
        "search.best_score_ "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "HbPjmGVZYE5f"
      },
      "outputs": [],
      "source": [
        "c=search.best_params_"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "n6df53JqvdJn"
      },
      "outputs": [],
      "source": [
        "# Fit the model on the entire training data using best hyperparameters\n",
        "lr = LinearRegression(fit_intercept=c[\"fit_intercept\"], copy_X=c[\"copy_X\"], positive=c[\"positive\"])\n",
        "lr.fit(x_train, y_train)\n",
        "\n",
        "# Predict the target variable for test data using the trained model\n",
        "y_pred = lr.predict(x_test)\n",
        "\n",
        "# Evaluate the model performance on test data\n",
        "from sklearn.metrics import r2_score, mean_squared_error\n",
        "\n",
        "print(\"R^2 score:\", r2_score(y_test, y_pred))\n",
        "print(\"MSE:\", mean_squared_error(y_test, y_pred))\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PiV4Ypx8fxKe"
      },
      "source": [
        "##### Which hyperparameter optimization technique have you used and why?"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "negyGRa7fxKf"
      },
      "source": [
        "Over here we have used **RandomizedSearchCV** hyperparameter optimization technique , the reason for using this technique is that In order to train and score the model, Random Search creates a grid of hyperparameter values and chooses random combinations. As a result, we are able to specifically regulate the quantity of parameter combinations that are tried. Based on available time or resources, the number of search iterations is decided."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TfvqoZmBfxKf"
      },
      "source": [
        "##### Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "OaLui8CcfxKf"
      },
      "source": [
        "We have seen slight fall in the accuracy of the model by using the hyperparameter tuning. "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dJ2tPlVmpsJ0"
      },
      "source": [
        "###ML Model - 2 Decision Tree Regressor\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JWYfwnehpsJ1"
      },
      "source": [
        "#### 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "yEl-hgQWpsJ1"
      },
      "outputs": [],
      "source": [
        "from sklearn.tree import DecisionTreeRegressor\n",
        "decision_tree_model = DecisionTreeRegressor()\n",
        "decision_tree_model.fit(x_train, y_train)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "I_gGJblmEsDW"
      },
      "outputs": [],
      "source": [
        "decision_tree_model.score(x_test, y_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "OMEtfwWEgLH7"
      },
      "outputs": [],
      "source": [
        "# predict the train and test data\n",
        "y_train_pred = decision_tree_model.predict(x_train)\n",
        "y_test_pred = decision_tree_model.predict(x_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7KlOPLI-gS4V"
      },
      "outputs": [],
      "source": [
        "model_evaluation = calculate_model_metrics(y_train, y_train_pred, y_test, y_test_pred)\n",
        "\n",
        "metrics_table.loc[len(metrics_table.index)] = ['Decision Tree', model_evaluation['Train_R2'], model_evaluation['Test_R2'], \n",
        "                                                         model_evaluation['Train_RMSE'], model_evaluation['Test_RMSE'], \n",
        "                                                         model_evaluation['Train_RMSPE'], model_evaluation['Test_RMSPE']]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "88y50_SSFR6Y"
      },
      "source": [
        "Over here we have used Decision tree regressor, Decision tree regression observes features of an object and trains a model in the structure of a tree to predict data in the future to produce meaningful continuous output. Continuous output means that the output/result is not discrete, i.e., it is not represented just by a discrete, known set of numbers or values."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "E_nvtWKEKg6A"
      },
      "outputs": [],
      "source": [
        "metrics_table.loc[1:,:]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-jK_YjpMpsJ2"
      },
      "source": [
        "#### 2. Cross- Validation & Hyperparameter Tuning"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "hm5corBHgonX"
      },
      "outputs": [],
      "source": [
        "from sklearn.model_selection import RandomizedSearchCV\n",
        "from scipy.stats import randint\n",
        "from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score\n",
        "from pylab import rcParams\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Dn0EOfS6psJ2"
      },
      "outputs": [],
      "source": [
        "# ML Model - 1 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)\n",
        "# Fit the Algorithm\n",
        "# Predict on the model\n",
        "# cross validation using k fold technique\n",
        "from sklearn.model_selection import cross_val_score\n",
        "for i in [3,5,10]:\n",
        "  score = cross_val_score(DecisionTreeRegressor(), X, y,cv=i)\n",
        "  print(np.mean(score))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1W1PfiIhE1Au"
      },
      "outputs": [],
      "source": [
        "#hyper parameter tuning using RandomizedSearchCV\n",
        "from sklearn.model_selection import RandomizedSearchCV\n",
        "from scipy.stats import randint\n",
        "\n",
        "dtr = DecisionTreeRegressor()\n",
        "\n",
        "param_distributions = {\"criterion\": [\"squared_error\", \"poisson\",\"friedman_mse\" ],\n",
        "                        \"splitter\": [\"best\", \"random\"],\n",
        "                       \"max_features\": [\"auto\", \"sqrt\", \"log2\"], \"max_depth\" : [10]}\n",
        "search = RandomizedSearchCV(dtr, param_distributions).fit(X, y)\n",
        "search.best_score_ "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "DPYPjll1Lf12"
      },
      "outputs": [],
      "source": [
        "search.best_params_"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HAih1iBOpsJ2"
      },
      "source": [
        "##### Which hyperparameter optimization technique have you used and why?"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9kBgjYcdpsJ2"
      },
      "source": [
        "Over here we have used RandomizedSearchCV hyperparameter optimization technique , the reason for using this technique is that In order to train and score the model, Random Search creates a grid of hyperparameter values and chooses random combinations. As a result, we are able to specifically regulate the quantity of parameter combinations that are tried. Based on available time or resources, the number of search iterations is decided."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zVGeBEFhpsJ2"
      },
      "source": [
        "##### Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "74yRdG6UpsJ3"
      },
      "source": [
        "We have seen slight fall in the accuracy of the model by using the hyperparameter tuning."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bmKjuQ-FpsJ3"
      },
      "source": [
        "#### 3. Explain each evaluation metric's indication towards business and the business impact pf the ML model used."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "BDKtOrBQpsJ3"
      },
      "source": [
        "Evaluation metrics are used in machine learning to measure the performance of a model on a given dataset. This allows us to compare the performance of different models and select the one that performs the best on the task at hand. we have used evaluation metrics for this tasks, mean squared error for regression problems , RMSE."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "SHaT7Aw5FwrK"
      },
      "source": [
        "###ML Model - 3 Random Forest Regressor"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ES3OZYRIGQYr"
      },
      "source": [
        "####1. Explain the ML Model used and it's performance using Evaluation metric Score Chart."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "b6kPzEcTF0lq"
      },
      "outputs": [],
      "source": [
        "from sklearn.ensemble import RandomForestRegressor\n",
        "random_forest_model = RandomForestRegressor()\n",
        "random_forest_model.fit(x_train, y_train)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "W2djhduWF6f5"
      },
      "outputs": [],
      "source": [
        "random_forest_model.score(x_test, y_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "UkTd5U64ihXA"
      },
      "outputs": [],
      "source": [
        "# predict the train and test data\n",
        "y_train_pred = random_forest_model.predict(x_train)\n",
        "y_test_pred = random_forest_model.predict(x_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "44wzKDrtirAe"
      },
      "outputs": [],
      "source": [
        "# model evaluation\n",
        "model_evaluation = calculate_model_metrics(y_train, y_train_pred, y_test, y_test_pred)\n",
        "\n",
        "# add metrics to metrics table\n",
        "metrics_table.loc[len(metrics_table.index)] = ['Random Forest', model_evaluation['Train_R2'], model_evaluation['Test_R2'], \n",
        "                                                                model_evaluation['Train_RMSE'], model_evaluation['Test_RMSE'], \n",
        "                                                                model_evaluation['Train_RMSPE'], model_evaluation['Test_RMSPE']]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-SKblYRtGAj6"
      },
      "source": [
        "A supervised learning technique called Random Forest Regression leverages the ensemble learning approach for regression. The ensemble learning method combines predictions from various machine learning algorithms to provide predictions that are more accurate than those from a single model."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Fze-IPXLpx6K"
      },
      "source": [
        "### ML Model - 4 Lasso and Ridge Regression (L1 and L2 Regularization)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "FFrSXAtrpx6M"
      },
      "outputs": [],
      "source": [
        "# ML Model - 3 Implementation\n",
        "# Fit the Algorithm\n",
        "# Predict on the model\n",
        "from sklearn.linear_model import Lasso, Ridge, ElasticNet\n",
        "lasso_model = Lasso()\n",
        "lasso_model.fit(x_train, y_train)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Hsg6pPYuoUh-"
      },
      "outputs": [],
      "source": [
        "lasso_model.score(x_test, y_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Kub8Z6H1Ycva"
      },
      "outputs": [],
      "source": [
        "#ridge REgression\n",
        "ridge_model = Ridge()\n",
        "ridge_model.fit(x_train, y_train)\n",
        "ridge_model.score(x_test, y_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "gLoChJUnkYK_"
      },
      "outputs": [],
      "source": [
        "# predict the train and test data\n",
        "y_train_pred = ridge_model.predict(x_train)\n",
        "y_test_pred = ridge_model.predict(x_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "1zx7h0IHklt1"
      },
      "outputs": [],
      "source": [
        "# model evaluation\n",
        "model_evaluation = calculate_model_metrics(y_train, y_train_pred, y_test, y_test_pred)\n",
        "\n",
        "# add metrics to metrics table\n",
        "metrics_table.loc[len(metrics_table.index)] = ['Ridge', model_evaluation['Train_R2'], model_evaluation['Test_R2'], \n",
        "                                                                model_evaluation['Train_RMSE'], model_evaluation['Test_RMSE'], \n",
        "                                                                model_evaluation['Train_RMSPE'], model_evaluation['Test_RMSPE']]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "hGAT6Az9Yflc",
        "outputId": "4f11abdf-9031-4857-d58e-52d22ab43da9"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "0.1577893849036287"
            ]
          },
          "execution_count": 90,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "#elasticnet regression\n",
        "elasticnet_model = ElasticNet()\n",
        "elasticnet_model.fit(x_train, y_train)\n",
        "elasticnet_model.score(x_test, y_test)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7AN1z2sKpx6M"
      },
      "source": [
        "#### 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8NTpXf9_lMay"
      },
      "source": [
        "Evaluation metrics are used in machine learning to measure the performance of a model on a given dataset. This allows us to compare the performance of different models and select the one that performs the best on the task at hand. we have used evaluation metrics for this tasks, mean squared error for regression problems , RMSE."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "ze4_m4zNOmnZ",
        "outputId": "e7e7b365-e9e9-41f7-d0ff-ce632209b759"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-2d24e992-3674-4c31-bf97-ac176238babb\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Regression_Model</th>\n",
              "      <th>Train_R2</th>\n",
              "      <th>Test_R2</th>\n",
              "      <th>Train_RMSE</th>\n",
              "      <th>Test_RMSE</th>\n",
              "      <th>Train_RMSPE</th>\n",
              "      <th>Test_RMSPE</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Ridge</td>\n",
              "      <td>0.685865</td>\n",
              "      <td>0.69063</td>\n",
              "      <td>0.557798</td>\n",
              "      <td>0.566384</td>\n",
              "      <td>-7078.512679</td>\n",
              "      <td>1796.866966</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-2d24e992-3674-4c31-bf97-ac176238babb')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-2d24e992-3674-4c31-bf97-ac176238babb button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-2d24e992-3674-4c31-bf97-ac176238babb');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "  Regression_Model  Train_R2  Test_R2  Train_RMSE  Test_RMSE  Train_RMSPE  \\\n",
              "3            Ridge  0.685865  0.69063    0.557798   0.566384 -7078.512679   \n",
              "\n",
              "    Test_RMSPE  \n",
              "3  1796.866966  "
            ]
          },
          "execution_count": 91,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "metrics_table.loc[3:,:]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9PIHJqyupx6M"
      },
      "source": [
        "#### 2. Cross- Validation & Hyperparameter Tuning"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "2J4cN8T_lMRp"
      },
      "outputs": [],
      "source": [
        "from sklearn.model_selection import GridSearchCV"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "eSVXuaSKpx6M",
        "outputId": "663e2cc5-25d5-43f8-8b3d-faa0057d2532"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "The best fit alpha value is found out to be : {'alpha': 1e-15}\n",
            "\n",
            "Using  {'alpha': 1e-15}  the negative mean squared error is:  -0.31423038494557215\n"
          ]
        }
      ],
      "source": [
        "# ML Model - 3 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)\n",
        "lasso = Lasso()\n",
        "parameters = {'alpha': [1e-15,1e-13,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1e-1,1,5,10]}\n",
        "lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=3)\n",
        "lasso_regressor.fit(x_train, y_train)\n",
        "# Predict on the model\n",
        "print(\"The best fit alpha value is found out to be :\" ,lasso_regressor.best_params_)\n",
        "print(\"\\nUsing \",lasso_regressor.best_params_, \" the negative mean squared error is: \", lasso_regressor.best_score_)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "hODqT4OEmxsD"
      },
      "outputs": [],
      "source": [
        "# predict the train and test data\n",
        "y_train_pred = lasso_regressor.predict(x_train)\n",
        "y_test_pred = lasso_regressor.predict(x_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "RzNUbJ4Nmywt"
      },
      "outputs": [],
      "source": [
        "# model evaluation\n",
        "model_evaluation = calculate_model_metrics(y_train, y_train_pred, y_test, y_test_pred)\n",
        "# add metrics to metrics table\n",
        "metrics_table.loc[len(metrics_table.index)] = ['Lasso', model_evaluation['Train_R2'], model_evaluation['Test_R2'], \n",
        "                                                        model_evaluation['Train_RMSE'], model_evaluation['Test_RMSE'], \n",
        "                                                        model_evaluation['Train_RMSPE'], model_evaluation['Test_RMSPE']]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_-qAgymDpx6N"
      },
      "source": [
        "##### Which hyperparameter optimization technique have you used and why?"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lQMffxkwpx6N"
      },
      "source": [
        "Over here we have used GridSearchCV hyperparameter optimization technique , the reason for using this technique is that In order to train and score the model, grid Search creates a grid of hyperparameter values and chooses random combinations. As a result, we are able to specifically regulate the quantity of parameter combinations that are tried. Based on available time or resources, the number of search iterations is decided "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Z-hykwinpx6N"
      },
      "source": [
        "##### Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MzVzZC6opx6N"
      },
      "source": [
        "We have seen highly jump in the accuracy of the model by using the hyperparameter tuning our lasso r2 score increase to .82"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "h_CCil-SKHpo"
      },
      "source": [
        "### 1. Which Evaluation metrics did you consider for a positive business impact and why?"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jHVz9hHDKFms"
      },
      "source": [
        "Evaluation metrics are used in machine learning to measure the performance of a model on a given dataset. This allows us to compare the performance of different models and select the one that performs the best on the task at hand. we have used evaluation metrics for this tasks, mean squared error for regression problems , RMSE.Answer Here."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "cBFFvTBNJzUa"
      },
      "source": [
        "### 2. Which ML model did you choose from the above created models as your final prediction model and why?"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6ksF5Q1LKTVm"
      },
      "source": [
        "As we have seen above that, Random Forest Regressor is performing the best with the accuracy of 97.3% followed by Decison Tree Regressor with accuracy of 94.3%\n",
        "so here we are choosing Random Forest Regressor for best prediction."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "DlJ88mOxoPgQ",
        "outputId": "2003100b-0161-4326-8416-3fedc30d9cf0"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-ec878028-125e-49b4-a74e-b369ee45e69f\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Regression_Model</th>\n",
              "      <th>Train_R2</th>\n",
              "      <th>Test_R2</th>\n",
              "      <th>Train_RMSE</th>\n",
              "      <th>Test_RMSE</th>\n",
              "      <th>Train_RMSPE</th>\n",
              "      <th>Test_RMSPE</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Linear</td>\n",
              "      <td>0.685865</td>\n",
              "      <td>0.690627</td>\n",
              "      <td>0.557798</td>\n",
              "      <td>0.566387</td>\n",
              "      <td>-7078.509734</td>\n",
              "      <td>1796.875320</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Decision Tree</td>\n",
              "      <td>0.916867</td>\n",
              "      <td>0.635236</td>\n",
              "      <td>0.286950</td>\n",
              "      <td>0.615004</td>\n",
              "      <td>-3641.420003</td>\n",
              "      <td>1951.115070</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Random Forest</td>\n",
              "      <td>0.892630</td>\n",
              "      <td>0.725976</td>\n",
              "      <td>0.326106</td>\n",
              "      <td>0.533047</td>\n",
              "      <td>-4138.321226</td>\n",
              "      <td>1691.105089</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Ridge</td>\n",
              "      <td>0.685865</td>\n",
              "      <td>0.690630</td>\n",
              "      <td>0.557798</td>\n",
              "      <td>0.566384</td>\n",
              "      <td>-7078.512679</td>\n",
              "      <td>1796.866966</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Lasso</td>\n",
              "      <td>0.685865</td>\n",
              "      <td>0.690627</td>\n",
              "      <td>0.557798</td>\n",
              "      <td>0.566387</td>\n",
              "      <td>-7078.509734</td>\n",
              "      <td>1796.875320</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-ec878028-125e-49b4-a74e-b369ee45e69f')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-ec878028-125e-49b4-a74e-b369ee45e69f button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-ec878028-125e-49b4-a74e-b369ee45e69f');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "  Regression_Model  Train_R2   Test_R2  Train_RMSE  Test_RMSE  Train_RMSPE  \\\n",
              "0           Linear  0.685865  0.690627    0.557798   0.566387 -7078.509734   \n",
              "1    Decision Tree  0.916867  0.635236    0.286950   0.615004 -3641.420003   \n",
              "2    Random Forest  0.892630  0.725976    0.326106   0.533047 -4138.321226   \n",
              "3            Ridge  0.685865  0.690630    0.557798   0.566384 -7078.512679   \n",
              "4            Lasso  0.685865  0.690627    0.557798   0.566387 -7078.509734   \n",
              "\n",
              "    Test_RMSPE  \n",
              "0  1796.875320  \n",
              "1  1951.115070  \n",
              "2  1691.105089  \n",
              "3  1796.866966  \n",
              "4  1796.875320  "
            ]
          },
          "execution_count": 96,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# print metrics table\n",
        "metrics_table"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HvGl1hHyA_VK"
      },
      "source": [
        "### 3. Explain the model which you have used and the feature importance using any model explainability tool?"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YnvVTiIxBL-C"
      },
      "source": [
        "Random Forest is a supervised learning algorithm that can be used for both classification and regression tasks. A Random Forest regressor is a specific type of Random Forest that is used for regression tasks, which involve predicting a continuous output value (such as a price or temperature) rather than a discrete class label.\n",
        "\n",
        "The algorithm works by creating an ensemble of decision trees, where each tree is trained on a random subset of the data. The final output is then obtained by averaging the predictions of all the trees. This helps to reduce overfitting and improve the overall performance of the model.\n",
        "\n",
        "Random Forest regressor is known to be a very powerful algorithm that can handle high-dimensional data and a large number of input features. It is also relatively easy to use and interpret. It has several parameters that can be adjusted to optimize its performance, such as the number of trees in the ensemble, the maximum depth of each tree, and the minimum number of samples required to split a node.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "t_vhQUb6oulP"
      },
      "source": [
        "### Here we are taking randomly 20 datapoints and predicting their price by the trained best model."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "DZWRtFMuafnr"
      },
      "outputs": [],
      "source": [
        "#Here we are randomly taking 20 data points\n",
        "sample_df = df3.sample(n=20)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "DibPQFXRacjm"
      },
      "outputs": [],
      "source": [
        "# Here we are creating the target and feature variable\n",
        "X, Y = sample_df.loc[:, sample_df.columns != 'selling_price'], sample_df['selling_price']"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "KX0PpckdbMTD",
        "outputId": "9244e78a-ee53-4c1e-a9a4-95414aa68d31"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "array([-1.57660121, -1.17422109, -1.19069802, -0.07342126, -0.4468702 ,\n",
              "       -0.00510648,  0.06188682, -0.32993257, -0.54622848, -1.31898942,\n",
              "        0.31575663, -0.31857034,  0.18098694,  0.39025914,  0.59577343,\n",
              "        0.77591391, -0.41359796,  0.67957124, -0.36671499, -1.0900339 ])"
            ]
          },
          "execution_count": 99,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "#predicted value\n",
        "predict_20 = random_forest_model.predict(X)\n",
        "predict_20"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "uRb_9qoS7UAV",
        "outputId": "675282b9-aa12-4947-c540-9ea9a0dfedd6"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "2780   -1.616623\n",
              "1120   -1.007895\n",
              "3543   -1.007895\n",
              "3736   -0.399167\n",
              "3610   -0.399167\n",
              "3034   -0.423241\n",
              "1979    0.001792\n",
              "2470   -0.222301\n",
              "1769    0.540403\n",
              "1143   -0.968821\n",
              "2480    0.593374\n",
              "1118    0.426826\n",
              "652    -0.307456\n",
              "3083    0.032048\n",
              "3202    0.473564\n",
              "2809    1.252819\n",
              "1870   -0.551500\n",
              "1422    0.784447\n",
              "4030   -0.104994\n",
              "2423   -1.007895\n",
              "Name: selling_price, dtype: float64"
            ]
          },
          "execution_count": 100,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Actual value\n",
        "Y"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "vA0pX3TMhyVk",
        "outputId": "ca7c08e1-193f-4854-eaa6-16eed8baca94"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "#plot the actual and predicted value\n",
        "plt.scatter(Y, predict_20)\n",
        "plt.xlabel(\"Actual Values\")\n",
        "plt.ylabel(\"Predicted Values\")\n",
        "plt.title(\"Actual vs. Predicted Values\")\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "qGDIna2Ne5Ql",
        "outputId": "148cc7ff-7771-440d-8096-0ba5978864e0"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "0.699671338180176"
            ]
          },
          "execution_count": 102,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "#finding the r2 score\n",
        "random_forest_model.score(X,Y)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "hDvRsx0CMuwW",
        "outputId": "5f52e606-c239-47c2-929e-50003613b99c"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-009f8e7c-f688-4ff6-a55d-00bc83d5bc9b\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Regression_Model</th>\n",
              "      <th>Train_R2</th>\n",
              "      <th>Test_R2</th>\n",
              "      <th>Train_RMSE</th>\n",
              "      <th>Test_RMSE</th>\n",
              "      <th>Train_RMSPE</th>\n",
              "      <th>Test_RMSPE</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Random Forest</td>\n",
              "      <td>0.892630</td>\n",
              "      <td>0.725976</td>\n",
              "      <td>0.326106</td>\n",
              "      <td>0.533047</td>\n",
              "      <td>-4138.321226</td>\n",
              "      <td>1691.105089</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Ridge</td>\n",
              "      <td>0.685865</td>\n",
              "      <td>0.690630</td>\n",
              "      <td>0.557798</td>\n",
              "      <td>0.566384</td>\n",
              "      <td>-7078.512679</td>\n",
              "      <td>1796.866966</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Lasso</td>\n",
              "      <td>0.685865</td>\n",
              "      <td>0.690627</td>\n",
              "      <td>0.557798</td>\n",
              "      <td>0.566387</td>\n",
              "      <td>-7078.509734</td>\n",
              "      <td>1796.875320</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-009f8e7c-f688-4ff6-a55d-00bc83d5bc9b')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-009f8e7c-f688-4ff6-a55d-00bc83d5bc9b button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-009f8e7c-f688-4ff6-a55d-00bc83d5bc9b');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "  Regression_Model  Train_R2   Test_R2  Train_RMSE  Test_RMSE  Train_RMSPE  \\\n",
              "2    Random Forest  0.892630  0.725976    0.326106   0.533047 -4138.321226   \n",
              "3            Ridge  0.685865  0.690630    0.557798   0.566384 -7078.512679   \n",
              "4            Lasso  0.685865  0.690627    0.557798   0.566387 -7078.509734   \n",
              "\n",
              "    Test_RMSPE  \n",
              "2  1691.105089  \n",
              "3  1796.866966  \n",
              "4  1796.875320  "
            ]
          },
          "execution_count": 103,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "metrics_table.loc[2:,:]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "DLx3iX6929oX",
        "outputId": "c3fd49ad-c4f8-451c-d8ed-cb1ed3bec7d8"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Index(['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type',\n",
              "       'transmission', 'owner'],\n",
              "      dtype='object')"
            ]
          },
          "execution_count": 104,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df.columns"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "bpfO2K1KqEiZ",
        "outputId": "fff3328a-4da0-486a-ad27-d0fc78ad8cb6"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Text(0.5, 0, 'Relative Importance')"
            ]
          },
          "execution_count": 105,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 576x720 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "features = X.columns\n",
        "importances = random_forest_model.feature_importances_\n",
        "indices = np.argsort(importances)\n",
        "plt.figure(figsize=(8,10))\n",
        "plt.title('Feature Importances', fontsize=20)\n",
        "plt.barh(range(len(indices)), importances[indices], color='red', align='center')\n",
        "plt.yticks(range(len(indices)), features[indices])\n",
        "plt.xlabel('Relative Importance')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ys8HaFFe7bsr"
      },
      "source": [
        "## ***6. Conclusion***"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vvUxYCoJ-yjb"
      },
      "source": [
        "\n",
        "\n",
        "*   Before creating the regression model, it is crucial to clean and pre-process the data so\n",
        " that it is in the right format.\n",
        "*   Unnecessary data that skew the results were also filtered out.\n",
        "\n",
        "\n",
        "*   Feature engineering and exploratory data analysis were performed to gather more meaningful information from the data.\n",
        "*   Apart from this, various data visualization, like box plot, frequency plot, histogram, pair plot, correlation matrix and scatter plot were created to understand the uni-variate distribution and multi-variate relationship of the data\n",
        "\n",
        "\n",
        "*   Majority of cars are Diesel (49.6%) and Petrol (48.9%)\n",
        "*   The number of unique car count is: 1491\n",
        "*   Maximum sold cars belongs to First owners.\n",
        "*   selling price of cars is decreasing as per their running status.\n",
        "*   we performed Regression Analysis on our data set to model the prices of cars\n",
        " \n",
        "**ML model**\n",
        "\n",
        "\n",
        "\n",
        "*   Among the all regression models, it is clear that Random Forest Regressor is giving the best result with the accuracy of 72.4% followed by linear Regressor with accuracy of 69.2%. So, we will use the random forest regressor to predict the sales.\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "OO0xa9c8-nNo"
      },
      "source": [
        "### ***Hurrah! You have successfully completed your Machine Learning Capstone Project !!!***"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}