{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Project: Identify Customer Segments\n",
    "\n",
    "In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.\n",
    "\n",
    "This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.\n",
    "\n",
    "It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.\n",
    "\n",
    "At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import libraries here; add more as necessary\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import ast\n",
    "from sklearn.preprocessing import Imputer, StandardScaler\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.cluster import KMeans\n",
    "from collections import Counter\n",
    "\n",
    "# magic word for producing visualizations in notebook\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 0: Load the Data\n",
    "\n",
    "There are four files associated with this project (not including this one):\n",
    "\n",
    "- `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).\n",
    "- `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).\n",
    "- `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.\n",
    "- `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns\n",
    "\n",
    "Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.\n",
    "\n",
    "To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.\n",
    "\n",
    "Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load in the general demographics data.\n",
    "azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv',sep=';')\n",
    "\n",
    "# Load in the feature summary file.\n",
    "feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv',sep=';')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(891221, 85)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>AGER_TYP</th>\n",
       "      <th>ALTERSKATEGORIE_GROB</th>\n",
       "      <th>ANREDE_KZ</th>\n",
       "      <th>CJT_GESAMTTYP</th>\n",
       "      <th>FINANZ_MINIMALIST</th>\n",
       "      <th>FINANZ_SPARER</th>\n",
       "      <th>FINANZ_VORSORGER</th>\n",
       "      <th>FINANZ_ANLEGER</th>\n",
       "      <th>FINANZ_UNAUFFAELLIGER</th>\n",
       "      <th>FINANZ_HAUSBAUER</th>\n",
       "      <th>...</th>\n",
       "      <th>PLZ8_ANTG1</th>\n",
       "      <th>PLZ8_ANTG2</th>\n",
       "      <th>PLZ8_ANTG3</th>\n",
       "      <th>PLZ8_ANTG4</th>\n",
       "      <th>PLZ8_BAUMAX</th>\n",
       "      <th>PLZ8_HHZ</th>\n",
       "      <th>PLZ8_GBZ</th>\n",
       "      <th>ARBEIT</th>\n",
       "      <th>ORTSGR_KLS9</th>\n",
       "      <th>RELAT_AB</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>5</td>\n",
       "      <td>5</td>\n",
       "      <td>3</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>...</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>5</td>\n",
       "      <td>...</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>2.0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>...</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>5.0</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>...</td>\n",
       "      <td>2.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 85 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   AGER_TYP  ALTERSKATEGORIE_GROB  ANREDE_KZ  CJT_GESAMTTYP  \\\n",
       "0        -1                     2          1            2.0   \n",
       "1        -1                     1          2            5.0   \n",
       "2        -1                     3          2            3.0   \n",
       "3         2                     4          2            2.0   \n",
       "4        -1                     3          1            5.0   \n",
       "\n",
       "   FINANZ_MINIMALIST  FINANZ_SPARER  FINANZ_VORSORGER  FINANZ_ANLEGER  \\\n",
       "0                  3              4                 3               5   \n",
       "1                  1              5                 2               5   \n",
       "2                  1              4                 1               2   \n",
       "3                  4              2                 5               2   \n",
       "4                  4              3                 4               1   \n",
       "\n",
       "   FINANZ_UNAUFFAELLIGER  FINANZ_HAUSBAUER    ...     PLZ8_ANTG1  PLZ8_ANTG2  \\\n",
       "0                      5                 3    ...            NaN         NaN   \n",
       "1                      4                 5    ...            2.0         3.0   \n",
       "2                      3                 5    ...            3.0         3.0   \n",
       "3                      1                 2    ...            2.0         2.0   \n",
       "4                      3                 2    ...            2.0         4.0   \n",
       "\n",
       "   PLZ8_ANTG3  PLZ8_ANTG4  PLZ8_BAUMAX  PLZ8_HHZ  PLZ8_GBZ  ARBEIT  \\\n",
       "0         NaN         NaN          NaN       NaN       NaN     NaN   \n",
       "1         2.0         1.0          1.0       5.0       4.0     3.0   \n",
       "2         1.0         0.0          1.0       4.0       4.0     3.0   \n",
       "3         2.0         0.0          1.0       3.0       4.0     2.0   \n",
       "4         2.0         1.0          2.0       3.0       3.0     4.0   \n",
       "\n",
       "   ORTSGR_KLS9  RELAT_AB  \n",
       "0          NaN       NaN  \n",
       "1          5.0       4.0  \n",
       "2          5.0       2.0  \n",
       "3          3.0       3.0  \n",
       "4          6.0       5.0  \n",
       "\n",
       "[5 rows x 85 columns]"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check the structure of the data after it's loaded (e.g. print the number of\n",
    "# rows and columns, print the first few rows).\n",
    "print(azdias.shape)\n",
    "azdias.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(85, 4)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>attribute</th>\n",
       "      <th>information_level</th>\n",
       "      <th>type</th>\n",
       "      <th>missing_or_unknown</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>AGER_TYP</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[-1,0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ALTERSKATEGORIE_GROB</td>\n",
       "      <td>person</td>\n",
       "      <td>ordinal</td>\n",
       "      <td>[-1,0,9]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>ANREDE_KZ</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[-1,0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>CJT_GESAMTTYP</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>FINANZ_MINIMALIST</td>\n",
       "      <td>person</td>\n",
       "      <td>ordinal</td>\n",
       "      <td>[-1]</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              attribute information_level         type missing_or_unknown\n",
       "0              AGER_TYP            person  categorical             [-1,0]\n",
       "1  ALTERSKATEGORIE_GROB            person      ordinal           [-1,0,9]\n",
       "2             ANREDE_KZ            person  categorical             [-1,0]\n",
       "3         CJT_GESAMTTYP            person  categorical                [0]\n",
       "4     FINANZ_MINIMALIST            person      ordinal               [-1]"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(feat_info.shape)\n",
    "feat_info.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. \n",
    "\n",
    "## Step 1: Preprocessing\n",
    "\n",
    "### Step 1.1: Assess Missing Data\n",
    "\n",
    "The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!\n",
    "\n",
    "#### Step 1.1.1: Convert Missing Value Codes to NaNs\n",
    "The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.\n",
    "\n",
    "**As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Naturally missing data porcentage is 6.464149613118342 %\n"
     ]
    }
   ],
   "source": [
    "# Identify missing or unknown data values and convert them to NaNs.\n",
    "\n",
    "# Identify naturally missing data porcentage\n",
    "element_count = 0\n",
    "for i in azdias.count():\n",
    "    element_count = element_count + i\n",
    "print (\"Naturally missing data porcentage is {} %\".format(100*(1-element_count/azdias.size)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Method to obtain list of 'missing' or 'unknown'\n",
    "def create_x_list(x):\n",
    "    x = x.strip('[]')\n",
    "    x_list = x.split(\",\")\n",
    "    i = 0\n",
    "    for value in x_list:\n",
    "        if(is_integer(value)):\n",
    "            x_list[i] = int(x_list[i]) \n",
    "        i = i + 1\n",
    "    x_list=[item for item in x_list if item != '']\n",
    "    return x_list"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Method to check if value is integer\n",
    "def is_integer(value):\n",
    "    try:\n",
    "        int(value)\n",
    "        return True\n",
    "    except:\n",
    "        return False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "# Convert data that matches a 'missing' or 'unknown' value code into NaN\n",
    "for column in azdias:\n",
    "    tmp = feat_info.loc[(feat_info['attribute'] == column)]['missing_or_unknown']\n",
    "    for item in tmp:\n",
    "        x_list = create_x_list(item)\n",
    "    for element in x_list:\n",
    "        azdias.loc[azdias[column] == element, column] = np.NAN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Missing data porcentage is 11.054139407027652 %\n"
     ]
    }
   ],
   "source": [
    "# Identify missing data porcentage after replacing 'missing' or 'unknown' value codes\n",
    "element_count = 0\n",
    "for i in azdias.count():\n",
    "    element_count = element_count + i\n",
    "print (\"Missing data porcentage is {} %\".format(100*(1-element_count/azdias.size)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 1.1.2: Assess Missing Data in Each Column\n",
    "\n",
    "How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)\n",
    "\n",
    "For the remaining features, are there any patterns in which columns have, or share, missing data?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Perform an assessment of how much missing data there is in each column of the\n",
    "# dataset.\n",
    "\n",
    "# Create dictionary of number of NaN elements for each column\n",
    "nan_dict = {}\n",
    "i=0\n",
    "for column in azdias:\n",
    "    #print(\"column {}\".format(i))\n",
    "    nan_count = len(azdias[column])-azdias[column].count()\n",
    "    nan_dict.update({column:nan_count})\n",
    "    i=i+1\n",
    "#print(nan_dict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fda0e3ec0b8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Investigate patterns in the amount of missing data in each column.\n",
    "plt.figure(figsize=(20,6))\n",
    "plt.hist(nan_dict.values(), color='g',bins=30)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fda0e4208d0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Print number of missing values per column\n",
    "plt.figure(figsize=(20,6))\n",
    "x=range(0,azdias.shape[1])\n",
    "plt.bar(x,nan_dict.values(), color='g')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['AGER_TYP', 'GEBURTSJAHR', 'TITEL_KZ', 'ALTER_HH', 'KK_KUNDENTYP', 'KBA05_BAUMAX']\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ALTERSKATEGORIE_GROB</th>\n",
       "      <th>ANREDE_KZ</th>\n",
       "      <th>CJT_GESAMTTYP</th>\n",
       "      <th>FINANZ_MINIMALIST</th>\n",
       "      <th>FINANZ_SPARER</th>\n",
       "      <th>FINANZ_VORSORGER</th>\n",
       "      <th>FINANZ_ANLEGER</th>\n",
       "      <th>FINANZ_UNAUFFAELLIGER</th>\n",
       "      <th>FINANZ_HAUSBAUER</th>\n",
       "      <th>FINANZTYP</th>\n",
       "      <th>...</th>\n",
       "      <th>PLZ8_ANTG1</th>\n",
       "      <th>PLZ8_ANTG2</th>\n",
       "      <th>PLZ8_ANTG3</th>\n",
       "      <th>PLZ8_ANTG4</th>\n",
       "      <th>PLZ8_BAUMAX</th>\n",
       "      <th>PLZ8_HHZ</th>\n",
       "      <th>PLZ8_GBZ</th>\n",
       "      <th>ARBEIT</th>\n",
       "      <th>ORTSGR_KLS9</th>\n",
       "      <th>RELAT_AB</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>...</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>...</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>...</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>...</td>\n",
       "      <td>2.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 79 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   ALTERSKATEGORIE_GROB  ANREDE_KZ  CJT_GESAMTTYP  FINANZ_MINIMALIST  \\\n",
       "0                   2.0        1.0            2.0                3.0   \n",
       "1                   1.0        2.0            5.0                1.0   \n",
       "2                   3.0        2.0            3.0                1.0   \n",
       "3                   4.0        2.0            2.0                4.0   \n",
       "4                   3.0        1.0            5.0                4.0   \n",
       "\n",
       "   FINANZ_SPARER  FINANZ_VORSORGER  FINANZ_ANLEGER  FINANZ_UNAUFFAELLIGER  \\\n",
       "0            4.0               3.0             5.0                    5.0   \n",
       "1            5.0               2.0             5.0                    4.0   \n",
       "2            4.0               1.0             2.0                    3.0   \n",
       "3            2.0               5.0             2.0                    1.0   \n",
       "4            3.0               4.0             1.0                    3.0   \n",
       "\n",
       "   FINANZ_HAUSBAUER  FINANZTYP    ...     PLZ8_ANTG1  PLZ8_ANTG2  PLZ8_ANTG3  \\\n",
       "0               3.0        4.0    ...            NaN         NaN         NaN   \n",
       "1               5.0        1.0    ...            2.0         3.0         2.0   \n",
       "2               5.0        1.0    ...            3.0         3.0         1.0   \n",
       "3               2.0        6.0    ...            2.0         2.0         2.0   \n",
       "4               2.0        5.0    ...            2.0         4.0         2.0   \n",
       "\n",
       "   PLZ8_ANTG4  PLZ8_BAUMAX  PLZ8_HHZ  PLZ8_GBZ  ARBEIT  ORTSGR_KLS9  RELAT_AB  \n",
       "0         NaN          NaN       NaN       NaN     NaN          NaN       NaN  \n",
       "1         1.0          1.0       5.0       4.0     3.0          5.0       4.0  \n",
       "2         0.0          1.0       4.0       4.0     3.0          5.0       2.0  \n",
       "3         0.0          1.0       3.0       4.0     2.0          3.0       3.0  \n",
       "4         1.0          2.0       3.0       3.0     4.0          6.0       5.0  \n",
       "\n",
       "[5 rows x 79 columns]"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Remove the outlier columns from the dataset. (You'll perform other data\n",
    "# engineering tasks such as re-encoding and imputation later.)\n",
    "\n",
    "cols=[]\n",
    "for key in nan_dict.keys():\n",
    "    if nan_dict[key]>200000:\n",
    "        cols.append(key)\n",
    "print(cols)\n",
    "azdias.drop(cols,axis=1,inplace=True)\n",
    "azdias.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Discussion 1.1.2: Assess Missing Data in Each Column\n",
    "\n",
    "As it could be observed in the previous histogram, the percentage of missing data is less than 15% in almost all columns.\n",
    "However, it was noticed that six columns('AGER_TYP', 'GEBURTSJAHR', 'TITEL_KZ', 'ALTER_HH', 'KK_KUNDENTYP', 'KBA05_BAUMAX') showed an extremely high percentage of missing values. These columns are treated as outliers(dropped from the dataset).\n",
    "Furthermore the following patterns were found in the data:\n",
    "- KKK REGIOTYP columns had the same amount of missing values (both related to region)\n",
    "- KBA05_ANTG1 KBA05_ANTG2 KBA05_ANTG3 KBA05_ANTG4 KBA05_GBZ MOBI_REGIO columns had the same amount of missing values (all related to living place)\n",
    "- PLZ8_ANTG1 PLZ8_ANTG2 PLZ8_ANTG3 PLZ8_ANTG4 PLZ8_BAUMAX PLZ8_GBZ PLZ8_HHZ columns had the same amount of missing values (all related to PLZ8 region living place info)\n",
    "- HEALTH_TYP SHOPPER_TYP VERS_TYP columns had the same amount of missing values (related to health/shopping)\n",
    "- CAMEO_DEUG_2015 CAMEO_DEU_2015 CAMEO_INTL_2015 columns had the same amount of missing values (related to micro-cell features)\n",
    "- ARBEIT RELAT_AB columns had the same amount of missing values (related to employment)\n",
    "- BALLRAUM EWDICHTE INNENSTADT columns had the same amount of missing values (related to postcode-level features)\n",
    "- GEBAEUDETYP MIN_GEBAEUDEJAHR OST_WEST_KZ WOHNLAGE columns had the same amount of missing values (related to building/neighborhood)\n",
    "- LP_FAMILIE_FEIN LP_FAMILIE_GROB columns had the same amount of missing values (related to family)\n",
    "- ANZ_PERSONEN ANZ_TITEL SOHO_KZ WOHNDAUER_2008 columns had the same amount of missing values (related to household information)\n",
    "- CJT_GESAMTTYP GFK_URLAUBERTYP LP_STATUS_FEIN LP_STATUS_GROB ONLINE_AFFINITAET RETOURTYP_BK_S columns had the same amount of missing values (related to consumer habits and social status)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 1.1.3: Assess Missing Data in Each Row\n",
    "\n",
    "Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.\n",
    "\n",
    "In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.\n",
    "- You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.\n",
    "- To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.\n",
    "\n",
    "Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. Make sure you report your observations in the discussion section. **Either way, you should continue your analysis below using just the subset of the data with few or no missing values.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "# How much data is missing in each row of the dataset?\n",
    "\n",
    "# Create list of number of NaN elements for each row\n",
    "nan_list = []\n",
    "rows=azdias.shape[0]\n",
    "columns=azdias.shape[1]\n",
    "for i in range(0,rows):\n",
    "    nan_count = columns-azdias.loc[i].count()\n",
    "    nan_list.append(nan_count)\n",
    "#print(nan_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fda0e321320>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Write code to divide the data into two subsets based on the number of missing\n",
    "# values in each row.\n",
    "\n",
    "# Plotting histogram of missing values per row\n",
    "plt.figure(figsize=(20,6))\n",
    "plt.hist(nan_list, color='g',bins=50)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find row numbers to be separated (divided)\n",
    "# Threshold: more than 20 missing elements\n",
    "i=0\n",
    "nan_list_row=[]\n",
    "for element in nan_list:\n",
    "    if(element>20):\n",
    "        nan_list_row.append(i)\n",
    "    i = i + 1\n",
    "#print(nan_list_row)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Divide dataframe\n",
    "azdias_nan_subset = pd.DataFrame(columns=azdias.columns)\n",
    "azdias_nan_subset = azdias_nan_subset.append(azdias.loc[nan_list_row],ignore_index=True)\n",
    "#azdias_nan_subset.shape\n",
    "\n",
    "azdias.drop(nan_list_row,inplace=True)\n",
    "#azdias.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compare the distribution of values for at least five columns where there are\n",
    "# no or few missing values, between the two subsets.\n",
    "\n",
    "# Function to plot values\n",
    "def compare_col_hist(col):\n",
    "    fig, axs = plt.subplots(1, 2, tight_layout=True)\n",
    "    azdias.hist(column=col,bins=50,ax=axs[0])\n",
    "    azdias_nan_subset.hist(column=col,bins=50,ax=axs[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ANREDE_KZ\n",
      "FINANZ_MINIMALIST\n",
      "FINANZ_SPARER\n",
      "FINANZ_VORSORGER\n",
      "FINANZ_ANLEGER\n",
      "FINANZ_UNAUFFAELLIGER\n",
      "FINANZ_HAUSBAUER\n",
      "FINANZTYP\n",
      "GREEN_AVANTGARDE\n",
      "SEMIO_SOZ\n",
      "SEMIO_FAM\n",
      "SEMIO_REL\n",
      "SEMIO_MAT\n",
      "SEMIO_VERT\n",
      "SEMIO_LUST\n",
      "SEMIO_ERL\n",
      "SEMIO_KULT\n",
      "SEMIO_RAT\n",
      "SEMIO_KRIT\n",
      "SEMIO_DOM\n",
      "SEMIO_KAEM\n",
      "SEMIO_PFLICHT\n",
      "SEMIO_TRADV\n",
      "ZABEOTYP\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.6/site-packages/matplotlib/figure.py:1999: UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.\n",
      "  warnings.warn(\"This figure includes Axes that are not compatible \"\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fda0e36ba20>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fda31cb1860>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fda0ee528d0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fda31de6eb8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fd9e33780f0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot and compare\n",
    "\n",
    "# Find columns with no nan-elements\n",
    "i=0\n",
    "for col in nan_dict.keys():\n",
    "    if(nan_dict[col]==0):\n",
    "        print(col)\n",
    "    i = i + 1\n",
    "        \n",
    "pick_columns = ['FINANZ_SPARER','SEMIO_LUST','SEMIO_LUST','SEMIO_TRADV','ZABEOTYP']\n",
    "for col in pick_columns:\n",
    "    compare_col_hist(col)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Discussion 1.1.3: Assess Missing Data in Each Row\n",
    "\n",
    "After plotting the number of missing values per row, it could be noticed that most rows had less that 20 missing values.\n",
    "Some rows had more than 20 missing values (those rows are treated as outliers).\n",
    "After dividing the data based on this threshold (20 values missing per row), and comparing the column value distribution for columns with zero missing values(5 samples), it could be noticed that:\n",
    "- All columns had different distributions. Special treatment may be required for these columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 1.2: Select and Re-Encode Features\n",
    "\n",
    "Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.\n",
    "- For numeric and interval data, these features can be kept without changes.\n",
    "- Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).\n",
    "- Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.\n",
    "\n",
    "In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.\n",
    "\n",
    "Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "type\n",
       "categorical    21\n",
       "interval        1\n",
       "mixed           7\n",
       "numeric         7\n",
       "ordinal        49\n",
       "dtype: int64"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# How many features are there of each data type?\n",
    "\n",
    "feat_info.groupby(['type']).size()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 1.2.1: Re-Encode Categorical Features\n",
    "\n",
    "For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:\n",
    "- For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.\n",
    "- There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.\n",
    "- For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>attribute</th>\n",
       "      <th>information_level</th>\n",
       "      <th>type</th>\n",
       "      <th>missing_or_unknown</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>AGER_TYP</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[-1,0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>ANREDE_KZ</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[-1,0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>CJT_GESAMTTYP</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>FINANZTYP</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[-1]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>GFK_URLAUBERTYP</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>GREEN_AVANTGARDE</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>LP_FAMILIE_FEIN</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>LP_FAMILIE_GROB</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>LP_STATUS_FEIN</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>LP_STATUS_GROB</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>NATIONALITAET_KZ</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[-1,0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>38</th>\n",
       "      <td>SHOPPER_TYP</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[-1]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39</th>\n",
       "      <td>SOHO_KZ</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[-1]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>40</th>\n",
       "      <td>TITEL_KZ</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[-1,0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>41</th>\n",
       "      <td>VERS_TYP</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[-1]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>42</th>\n",
       "      <td>ZABEOTYP</td>\n",
       "      <td>person</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[-1,9]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>47</th>\n",
       "      <td>KK_KUNDENTYP</td>\n",
       "      <td>household</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[-1]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>52</th>\n",
       "      <td>GEBAEUDETYP</td>\n",
       "      <td>building</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[-1,0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>55</th>\n",
       "      <td>OST_WEST_KZ</td>\n",
       "      <td>building</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[-1]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>57</th>\n",
       "      <td>CAMEO_DEUG_2015</td>\n",
       "      <td>microcell_rr4</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[-1,X]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>58</th>\n",
       "      <td>CAMEO_DEU_2015</td>\n",
       "      <td>microcell_rr4</td>\n",
       "      <td>categorical</td>\n",
       "      <td>[XX]</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           attribute information_level         type missing_or_unknown\n",
       "0           AGER_TYP            person  categorical             [-1,0]\n",
       "2          ANREDE_KZ            person  categorical             [-1,0]\n",
       "3      CJT_GESAMTTYP            person  categorical                [0]\n",
       "10         FINANZTYP            person  categorical               [-1]\n",
       "12   GFK_URLAUBERTYP            person  categorical                 []\n",
       "13  GREEN_AVANTGARDE            person  categorical                 []\n",
       "17   LP_FAMILIE_FEIN            person  categorical                [0]\n",
       "18   LP_FAMILIE_GROB            person  categorical                [0]\n",
       "19    LP_STATUS_FEIN            person  categorical                [0]\n",
       "20    LP_STATUS_GROB            person  categorical                [0]\n",
       "21  NATIONALITAET_KZ            person  categorical             [-1,0]\n",
       "38       SHOPPER_TYP            person  categorical               [-1]\n",
       "39           SOHO_KZ            person  categorical               [-1]\n",
       "40          TITEL_KZ            person  categorical             [-1,0]\n",
       "41          VERS_TYP            person  categorical               [-1]\n",
       "42          ZABEOTYP            person  categorical             [-1,9]\n",
       "47      KK_KUNDENTYP         household  categorical               [-1]\n",
       "52       GEBAEUDETYP          building  categorical             [-1,0]\n",
       "55       OST_WEST_KZ          building  categorical               [-1]\n",
       "57   CAMEO_DEUG_2015     microcell_rr4  categorical             [-1,X]\n",
       "58    CAMEO_DEU_2015     microcell_rr4  categorical               [XX]"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Assess categorical variables: which are binary, which are multi-level, and\n",
    "# which one needs to be re-encoded?\n",
    "\n",
    "feat_info[feat_info['type']=='categorical']\n",
    "#azdias[azdias.columns[53]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "more than 2 categories: CJT_GESAMTTYP\n",
      "more than 2 categories: FINANZTYP\n",
      "more than 2 categories: GFK_URLAUBERTYP\n",
      "more than 2 categories: LP_FAMILIE_FEIN\n",
      "more than 2 categories: LP_FAMILIE_GROB\n",
      "more than 2 categories: LP_STATUS_FEIN\n",
      "more than 2 categories: LP_STATUS_GROB\n",
      "more than 2 categories: NATIONALITAET_KZ\n",
      "more than 2 categories: SHOPPER_TYP\n",
      "more than 2 categories: VERS_TYP\n",
      "more than 2 categories: ZABEOTYP\n",
      "more than 2 categories: GEBAEUDETYP\n",
      "transformed by get_dummies: OST_WEST_KZ\n",
      "more than 2 categories: CAMEO_DEUG_2015\n",
      "more than 2 categories: CAMEO_DEU_2015\n"
     ]
    }
   ],
   "source": [
    "# Re-encode categorical variable(s) to be kept in the analysis.\n",
    "\n",
    "categorical_cols = []\n",
    "for a in feat_info[feat_info['type']=='categorical']['attribute']:\n",
    "    categorical_cols.append(a)\n",
    "    \n",
    "for cols in azdias.columns:\n",
    "    if cols in categorical_cols:\n",
    "        if len(azdias[cols].unique()) > 2:\n",
    "            azdias = azdias.drop(cols, axis=1)\n",
    "            print(\"more than 2 categories: {}\".format(cols))\n",
    "        else:\n",
    "            if(not is_integer(azdias[cols].unique()[0])):\n",
    "                dummies = pd.get_dummies(azdias[cols],prefix=cols)\n",
    "                azdias = azdias.drop(cols, axis=1)\n",
    "                azdias = azdias.join(dummies)\n",
    "                print(\"transformed by get_dummies: {}\".format(cols))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Discussion 1.2.1: Re-Encode Categorical Features\n",
    "\n",
    "feat_info has 21 categorical features. Out of them, 18 were present in our dataframe(the other three were previosuly deleted for having too many missing values). The categorical features with more than 3 values were deleted for simplicity. The binary categories were left untouched, except by OST_WEST_KZ. OST_WEST_KZ which was one-hot-encoded using get_dummies."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 1.2.2: Engineer Mixed-Type Features\n",
    "\n",
    "There are a handful of features that are marked as \"mixed\" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:\n",
    "- \"PRAEGENDE_JUGENDJAHRE\" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.\n",
    "- \"CAMEO_INTL_2015\" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).\n",
    "- If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.\n",
    "\n",
    "Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>attribute</th>\n",
       "      <th>information_level</th>\n",
       "      <th>type</th>\n",
       "      <th>missing_or_unknown</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>LP_LEBENSPHASE_FEIN</td>\n",
       "      <td>person</td>\n",
       "      <td>mixed</td>\n",
       "      <td>[0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>LP_LEBENSPHASE_GROB</td>\n",
       "      <td>person</td>\n",
       "      <td>mixed</td>\n",
       "      <td>[0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>PRAEGENDE_JUGENDJAHRE</td>\n",
       "      <td>person</td>\n",
       "      <td>mixed</td>\n",
       "      <td>[-1,0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>56</th>\n",
       "      <td>WOHNLAGE</td>\n",
       "      <td>building</td>\n",
       "      <td>mixed</td>\n",
       "      <td>[-1]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59</th>\n",
       "      <td>CAMEO_INTL_2015</td>\n",
       "      <td>microcell_rr4</td>\n",
       "      <td>mixed</td>\n",
       "      <td>[-1,XX]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>64</th>\n",
       "      <td>KBA05_BAUMAX</td>\n",
       "      <td>microcell_rr3</td>\n",
       "      <td>mixed</td>\n",
       "      <td>[-1,0]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>79</th>\n",
       "      <td>PLZ8_BAUMAX</td>\n",
       "      <td>macrocell_plz8</td>\n",
       "      <td>mixed</td>\n",
       "      <td>[-1,0]</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                attribute information_level   type missing_or_unknown\n",
       "15    LP_LEBENSPHASE_FEIN            person  mixed                [0]\n",
       "16    LP_LEBENSPHASE_GROB            person  mixed                [0]\n",
       "22  PRAEGENDE_JUGENDJAHRE            person  mixed             [-1,0]\n",
       "56               WOHNLAGE          building  mixed               [-1]\n",
       "59        CAMEO_INTL_2015     microcell_rr4  mixed            [-1,XX]\n",
       "64           KBA05_BAUMAX     microcell_rr3  mixed             [-1,0]\n",
       "79            PLZ8_BAUMAX    macrocell_plz8  mixed             [-1,0]"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "feat_info[feat_info['type']=='mixed']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Investigate \"PRAEGENDE_JUGENDJAHRE\" and engineer two new variables.\n",
    "\n",
    "PRAEGENDE_JUGENDJAHRE_new = azdias[['PRAEGENDE_JUGENDJAHRE', 'PRAEGENDE_JUGENDJAHRE']].copy()\n",
    "PRAEGENDE_JUGENDJAHRE_new.columns = ['PRAEGENDE_JUGENDJAHRE_DECADE','PRAEGENDE_JUGENDJAHRE_MOVEMENT']\n",
    "\n",
    "#Set Decade\n",
    "PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_DECADE'].isin([1,2]), 'PRAEGENDE_JUGENDJAHRE_DECADE'] = 40.\n",
    "PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_DECADE'].isin([3,4]), 'PRAEGENDE_JUGENDJAHRE_DECADE'] = 50.\n",
    "PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_DECADE'].isin([5,6,7]), 'PRAEGENDE_JUGENDJAHRE_DECADE'] = 60.\n",
    "PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_DECADE'].isin([8,9]), 'PRAEGENDE_JUGENDJAHRE_DECADE'] = 70.\n",
    "PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_DECADE'].isin([10,11,12,13]), 'PRAEGENDE_JUGENDJAHRE_DECADE'] = 80.\n",
    "PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_DECADE'].isin([14,15]), 'PRAEGENDE_JUGENDJAHRE_DECADE'] = 90.\n",
    "\n",
    "#Set Movement\n",
    "PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_MOVEMENT'].isin([1,3,5,8,10,12,14]), 'PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = 0.\n",
    "PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_MOVEMENT'].isin([2,4,6,7,9,11,13,15]), 'PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = 1. \n",
    "\n",
    "#Join Columns\n",
    "azdias = azdias.drop('PRAEGENDE_JUGENDJAHRE', axis=1)\n",
    "azdias = azdias.join(PRAEGENDE_JUGENDJAHRE_new)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Investigate \"CAMEO_INTL_2015\" and engineer two new variables.\n",
    "\n",
    "CAMEO_INTL_2015_new = azdias[['CAMEO_INTL_2015', 'CAMEO_INTL_2015']].copy()\n",
    "CAMEO_INTL_2015_new.columns = ['CAMEO_INTL_2015_WEALTH','CAMEO_INTL_2015_LIFE_STAGE']\n",
    "\n",
    "#Set Wealth\n",
    "CAMEO_INTL_2015_new['CAMEO_INTL_2015_WEALTH'] = round((CAMEO_INTL_2015_new['CAMEO_INTL_2015_WEALTH'].astype(float))/10)\n",
    "\n",
    "#Set Life Stage\n",
    "CAMEO_INTL_2015_new['CAMEO_INTL_2015_LIFE_STAGE'] = (CAMEO_INTL_2015_new['CAMEO_INTL_2015_LIFE_STAGE'].astype(float))%10\n",
    "\n",
    "#Join Columns\n",
    "azdias = azdias.drop('CAMEO_INTL_2015', axis=1)\n",
    "azdias = azdias.join(CAMEO_INTL_2015_new)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dropping the remaining 5 features (KBA05_BAUMAX already dropped)\n",
    "\n",
    "mixed_remaining_featues = ['LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB', 'WOHNLAGE', 'PLZ8_BAUMAX']\n",
    "azdias.drop(mixed_remaining_featues,axis=1,inplace=True)\n",
    "#azdias.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Discussion 1.2.2: Engineer Mixed-Type Features\n",
    "\n",
    "There are 7 mixed variables according to feat_info. 2 of them were re-engineered and the remaining 5 were dropped:\n",
    "- Two variables (PRAEGENDE_JUGENDJAHRE_MOVEMENT and PRAEGENDE_JUGENDJAHRE_DECADE) were created from the original PRAEGENDE_JUGENDJAHRE column based on Data_Dictionary.md\n",
    "- Two variables (CAMEO_INTL_2015_WEALTH and CAMEO_INTL_2015_LIFE_STAGE) were created from the original CAMEO_INTL_2015 column based on Data_Dictionary.md\n",
    "\n",
    "The remaining 5 variables were dropped folowwing the next criteria:\n",
    "- LP_LEBENSPHASE_FEIN LP_LEBENSPHASE_GROB WOHNLAGE : information provided by this features can be found in CAMEO_INTL_2015\n",
    "- PLZ8_BAUMAX : information provided by this features can be found in other PLZ8 macro-cell features\n",
    "- KBA05_BAUMAX : this feature was already dropped in the previous step"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 1.2.3: Complete Feature Selection\n",
    "\n",
    "In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:\n",
    "- All numeric, interval, and ordinal type columns from the original dataset.\n",
    "- Binary categorical features (all numerically-encoded).\n",
    "- Engineered features from other multi-level categorical features and mixed features.\n",
    "\n",
    "Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep \"PRAEGENDE_JUGENDJAHRE\", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "# If there are other re-engineering tasks you need to perform, make sure you\n",
    "# take care of them here. (Dealing with missing data will come in step 2.1.)\n",
    "\n",
    "# No more tasks needed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['ALTERSKATEGORIE_GROB', 'ANREDE_KZ', 'FINANZ_MINIMALIST',\n",
       "       'FINANZ_SPARER', 'FINANZ_VORSORGER', 'FINANZ_ANLEGER',\n",
       "       'FINANZ_UNAUFFAELLIGER', 'FINANZ_HAUSBAUER', 'GREEN_AVANTGARDE',\n",
       "       'HEALTH_TYP', 'RETOURTYP_BK_S', 'SEMIO_SOZ', 'SEMIO_FAM', 'SEMIO_REL',\n",
       "       'SEMIO_MAT', 'SEMIO_VERT', 'SEMIO_LUST', 'SEMIO_ERL', 'SEMIO_KULT',\n",
       "       'SEMIO_RAT', 'SEMIO_KRIT', 'SEMIO_DOM', 'SEMIO_KAEM', 'SEMIO_PFLICHT',\n",
       "       'SEMIO_TRADV', 'SOHO_KZ', 'ANZ_PERSONEN', 'ANZ_TITEL',\n",
       "       'HH_EINKOMMEN_SCORE', 'W_KEIT_KIND_HH', 'WOHNDAUER_2008',\n",
       "       'ANZ_HAUSHALTE_AKTIV', 'ANZ_HH_TITEL', 'KONSUMNAEHE',\n",
       "       'MIN_GEBAEUDEJAHR', 'KBA05_ANTG1', 'KBA05_ANTG2', 'KBA05_ANTG3',\n",
       "       'KBA05_ANTG4', 'KBA05_GBZ', 'BALLRAUM', 'EWDICHTE', 'INNENSTADT',\n",
       "       'GEBAEUDETYP_RASTER', 'KKK', 'MOBI_REGIO', 'ONLINE_AFFINITAET',\n",
       "       'REGIOTYP', 'KBA13_ANZAHL_PKW', 'PLZ8_ANTG1', 'PLZ8_ANTG2',\n",
       "       'PLZ8_ANTG3', 'PLZ8_ANTG4', 'PLZ8_HHZ', 'PLZ8_GBZ', 'ARBEIT',\n",
       "       'ORTSGR_KLS9', 'RELAT_AB', 'OST_WEST_KZ_O', 'OST_WEST_KZ_W',\n",
       "       'PRAEGENDE_JUGENDJAHRE_DECADE', 'PRAEGENDE_JUGENDJAHRE_MOVEMENT',\n",
       "       'CAMEO_INTL_2015_WEALTH', 'CAMEO_INTL_2015_LIFE_STAGE'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Do whatever you need to in order to ensure that the dataframe only contains\n",
    "# the columns that should be passed to the algorithm functions.\n",
    "\n",
    "# Check columns\n",
    "azdias.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 1.3: Create a Cleaning Function\n",
    "\n",
    "Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_data(df):\n",
    "    \"\"\"\n",
    "    Perform feature trimming, re-encoding, and engineering for demographics\n",
    "    data\n",
    "    \n",
    "    INPUT: Demographics DataFrame\n",
    "    OUTPUT: Trimmed and cleaned demographics DataFrame\n",
    "    \"\"\"\n",
    "    \n",
    "    # Put in code here to execute all main cleaning steps:\n",
    "    # convert missing value codes into NaNs, ...\n",
    "    \n",
    "    for column in df:\n",
    "        tmp = feat_info.loc[(feat_info['attribute'] == column)]['missing_or_unknown']\n",
    "        for item in tmp:\n",
    "            x_list = create_x_list(item)\n",
    "        for element in x_list:\n",
    "            df.loc[df[column] == element, column] = np.NAN\n",
    "    \n",
    "    # remove selected columns and rows, ...\n",
    "    rows=df.shape[0]\n",
    "    columns=df.shape[1]\n",
    "    cols = ['AGER_TYP', 'GEBURTSJAHR', 'TITEL_KZ', 'ALTER_HH', 'KK_KUNDENTYP', 'KBA05_BAUMAX']\n",
    "    #cols = []\n",
    "    #for column in df:\n",
    "    #    nan_count = columns-df[column].count()\n",
    "    #    if(nan_count>200000):\n",
    "    #        cols.append(column)\n",
    "    df.drop(cols,axis=1,inplace=True)\n",
    "    \n",
    "    rs = []\n",
    "    for row in range(0,rows):\n",
    "        nan_count = columns-df.loc[row].count()\n",
    "        if(nan_count>20):\n",
    "            rs.append(row)\n",
    "    df.drop(rs,inplace=True)\n",
    "    \n",
    "    # select, re-encode, and engineer column values.\n",
    "    # caterical values\n",
    "    categorical_cols = []\n",
    "    for a in feat_info[feat_info['type']=='categorical']['attribute']:\n",
    "        categorical_cols.append(a)\n",
    "    for cols in df.columns:\n",
    "        if cols in categorical_cols:\n",
    "            if len(df[cols].unique()) > 2:\n",
    "                df = df.drop(cols, axis=1)\n",
    "            else:\n",
    "                if(not is_integer(df[cols].unique()[0])):\n",
    "                    dummies = pd.get_dummies(df[cols],prefix=cols)\n",
    "                    df = df.drop(cols, axis=1)\n",
    "                    df = df.join(dummies)\n",
    "    \n",
    "    # mixed values\n",
    "    #\"PRAEGENDE_JUGENDJAHRE\" \n",
    "    PRAEGENDE_JUGENDJAHRE_new = df[['PRAEGENDE_JUGENDJAHRE', 'PRAEGENDE_JUGENDJAHRE']].copy()\n",
    "    PRAEGENDE_JUGENDJAHRE_new.columns = ['PRAEGENDE_JUGENDJAHRE_DECADE','PRAEGENDE_JUGENDJAHRE_MOVEMENT']\n",
    "    PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_DECADE'].isin([1,2]), 'PRAEGENDE_JUGENDJAHRE_DECADE'] = 40.\n",
    "    PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_DECADE'].isin([3,4]), 'PRAEGENDE_JUGENDJAHRE_DECADE'] = 50.\n",
    "    PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_DECADE'].isin([5,6,7]), 'PRAEGENDE_JUGENDJAHRE_DECADE'] = 60.\n",
    "    PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_DECADE'].isin([8,9]), 'PRAEGENDE_JUGENDJAHRE_DECADE'] = 70.\n",
    "    PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_DECADE'].isin([10,11,12,13]), 'PRAEGENDE_JUGENDJAHRE_DECADE'] = 80.\n",
    "    PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_DECADE'].isin([14,15]), 'PRAEGENDE_JUGENDJAHRE_DECADE'] = 90.\n",
    "    PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_MOVEMENT'].isin([1,3,5,8,10,12,14]), 'PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = 0.\n",
    "    PRAEGENDE_JUGENDJAHRE_new.loc[PRAEGENDE_JUGENDJAHRE_new['PRAEGENDE_JUGENDJAHRE_MOVEMENT'].isin([2,4,6,7,9,11,13,15]), 'PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = 1. \n",
    "    df = df.drop('PRAEGENDE_JUGENDJAHRE', axis=1)\n",
    "    df = df.join(PRAEGENDE_JUGENDJAHRE_new)\n",
    "    \n",
    "    #CAMEO_INTL_2015\n",
    "    CAMEO_INTL_2015_new = df[['CAMEO_INTL_2015', 'CAMEO_INTL_2015']].copy()\n",
    "    CAMEO_INTL_2015_new.columns = ['CAMEO_INTL_2015_WEALTH','CAMEO_INTL_2015_LIFE_STAGE']\n",
    "    CAMEO_INTL_2015_new['CAMEO_INTL_2015_WEALTH'] = round((CAMEO_INTL_2015_new['CAMEO_INTL_2015_WEALTH'].astype(float))/10)\n",
    "    CAMEO_INTL_2015_new['CAMEO_INTL_2015_LIFE_STAGE'] = (CAMEO_INTL_2015_new['CAMEO_INTL_2015_LIFE_STAGE'].astype(float))%10\n",
    "    df = df.drop('CAMEO_INTL_2015', axis=1)\n",
    "    df = df.join(CAMEO_INTL_2015_new)\n",
    "    \n",
    "    #Remaining mixed-type features\n",
    "    mixed_remaining_featues = ['LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB', 'WOHNLAGE', 'PLZ8_BAUMAX']\n",
    "    df = df.drop(mixed_remaining_featues,axis=1)\n",
    "\n",
    "    # Return the cleaned dataframe.\n",
    "    return df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Feature Transformation\n",
    "\n",
    "### Step 2.1: Apply Feature Scaling\n",
    "\n",
    "Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:\n",
    "\n",
    "- sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values before applying your scaler. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.\n",
    "- For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.\n",
    "- For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "# If you've not yet cleaned the dataset of all NaN values, then investigate and do that now.\n",
    "\n",
    "# Applying Imputer\n",
    "\n",
    "fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=1)\n",
    "azdias_imputed = pd.DataFrame(fill_NaN.fit_transform(azdias))\n",
    "azdias_imputed.columns = azdias.columns\n",
    "azdias_imputed.index = azdias.index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Apply feature scaling to the general population demographics data.\n",
    "\n",
    "scaler = StandardScaler()\n",
    "azdias_scaled = scaler.fit_transform(azdias_imputed.values)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Discussion 2.1: Apply Feature Scaling\n",
    "\n",
    "- Imputer was used to handle remaining NaN values\n",
    "- StandardScaler was used to scale the imputed data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2.2: Perform Dimensionality Reduction\n",
    "\n",
    "On your scaled data, you are now ready to apply dimensionality reduction techniques.\n",
    "\n",
    "- Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).\n",
    "- Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.\n",
    "- Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,\n",
       "  svd_solver='auto', tol=0.0, whiten=False)"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Apply PCA to the data.\n",
    "\n",
    "pca = PCA()\n",
    "pca.fit(azdias_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "def scree_plot(pca):\n",
    "    '''\n",
    "    Creates a scree plot associated with the principal components \n",
    "    \n",
    "    INPUT: pca - the result of instantian of PCA in scikit learn\n",
    "            \n",
    "    OUTPUT:\n",
    "            None\n",
    "    '''\n",
    "    num_components = len(pca.explained_variance_ratio_)\n",
    "    ind = np.arange(num_components)\n",
    "    vals = pca.explained_variance_ratio_\n",
    " \n",
    "    plt.figure(figsize=(20, 6))\n",
    "    ax = plt.subplot(111)\n",
    "    cumvals = np.cumsum(vals)\n",
    "    ax.bar(ind, vals)\n",
    "    ax.plot(ind, cumvals)\n",
    "    for i in range(num_components):\n",
    "        ax.annotate(r\"%s%%\" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va=\"bottom\", ha=\"center\", fontsize=12)\n",
    " \n",
    "    ax.xaxis.set_tick_params(width=0)\n",
    "    ax.yaxis.set_tick_params(width=2, length=12)\n",
    " \n",
    "    ax.set_xlabel(\"Principal Component\")\n",
    "    ax.set_ylabel(\"Variance Explained (%)\")\n",
    "    plt.title('Explained Variance Per Principal Component')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fda0e9825c0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Investigate the variance accounted for by each principal component.\n",
    "scree_plot(pca)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Re-apply PCA to the data while selecting for number of components to retain.\n",
    "\n",
    "# We can see that by using 45 components, the incremental variance explained variation ratio is near zero\n",
    "pca = PCA(n_components=45)\n",
    "azdias_scaled_select=pca.fit_transform(azdias_scaled)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Discussion 2.2: Perform Dimensionality Reduction\n",
    "\n",
    "It was found that by using 45 components, the incremental variance explained variation ratio became almost zero. Therefore, we will keep those 45 component for the upcoming analysis."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2.3: Interpret Principal Components\n",
    "\n",
    "Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.\n",
    "\n",
    "As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.\n",
    "\n",
    "- To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.\n",
    "- You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find feature weights for component. Return list of features sorted in ascendent order of weight\n",
    "def weights(component):\n",
    "    weight_dict={}\n",
    "    i = np.identity(azdias_scaled.shape[1]) \n",
    "    coef = pca.transform(i)\n",
    "    j=0\n",
    "    for col in azdias.columns:\n",
    "        weight_dict.update({col: coef[j][component]})\n",
    "        j = j + 1\n",
    "    final = sorted(weight_dict, key=weight_dict.get)\n",
    "    for element in final:\n",
    "        print(element,weight_dict[element])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "FINANZ_HAUSBAUER -0.102523681121\n",
      "HH_EINKOMMEN_SCORE -0.0808174612488\n",
      "SEMIO_LUST -0.0579878604962\n",
      "WOHNDAUER_2008 -0.0480758379478\n",
      "ANZ_HAUSHALTE_AKTIV -0.0383755335125\n",
      "EWDICHTE -0.0363816009634\n",
      "FINANZ_VORSORGER -0.0361934662805\n",
      "ORTSGR_KLS9 -0.0356993991414\n",
      "SEMIO_ERL -0.0327040269286\n",
      "SEMIO_DOM -0.0296753682843\n",
      "SEMIO_KAEM -0.021997838881\n",
      "RELAT_AB -0.0215800095676\n",
      "ARBEIT -0.0195112907643\n",
      "ANREDE_KZ -0.0163453773342\n",
      "ALTERSKATEGORIE_GROB -0.0126756088495\n",
      "SEMIO_SOZ -0.0120332908199\n",
      "OST_WEST_KZ_O -0.0119621687049\n",
      "HEALTH_TYP -0.00930481825689\n",
      "RETOURTYP_BK_S -0.00796378957142\n",
      "W_KEIT_KIND_HH -0.00344318076483\n",
      "ANZ_TITEL -0.000835163618243\n",
      "SOHO_KZ 0.000840817581233\n",
      "ANZ_HH_TITEL 0.00382693617738\n",
      "SEMIO_KRIT 0.00607886848819\n",
      "KBA13_ANZAHL_PKW 0.00673224135783\n",
      "PRAEGENDE_JUGENDJAHRE_MOVEMENT 0.00783649919684\n",
      "GREEN_AVANTGARDE 0.00925524685058\n",
      "SEMIO_KULT 0.0113327714166\n",
      "OST_WEST_KZ_W 0.0119621687049\n",
      "ANZ_PERSONEN 0.0128675970457\n",
      "SEMIO_PFLICHT 0.0177189976931\n",
      "SEMIO_REL 0.0220174858998\n",
      "FINANZ_SPARER 0.0238492801967\n",
      "FINANZ_ANLEGER 0.0239382496802\n",
      "SEMIO_FAM 0.0241607720285\n",
      "ONLINE_AFFINITAET 0.025175035698\n",
      "SEMIO_RAT 0.0259260229898\n",
      "BALLRAUM 0.0267315710146\n",
      "GEBAEUDETYP_RASTER 0.0308879980346\n",
      "SEMIO_TRADV 0.0312714517744\n",
      "CAMEO_INTL_2015_WEALTH 0.0330370781116\n",
      "INNENSTADT 0.0342543706364\n",
      "FINANZ_MINIMALIST 0.0355511784089\n",
      "SEMIO_VERT 0.0412617614064\n",
      "PRAEGENDE_JUGENDJAHRE_DECADE 0.044408711023\n",
      "SEMIO_MAT 0.0456797773756\n",
      "KONSUMNAEHE 0.0460153055338\n",
      "CAMEO_INTL_2015_LIFE_STAGE 0.0593140012579\n",
      "FINANZ_UNAUFFAELLIGER 0.0746070987525\n",
      "REGIOTYP 0.161705196623\n",
      "KKK 0.163408370492\n",
      "PLZ8_ANTG3 0.242789828441\n",
      "PLZ8_ANTG2 0.24463913351\n",
      "MIN_GEBAEUDEJAHR 0.244738823227\n",
      "PLZ8_ANTG4 0.244992596653\n",
      "PLZ8_HHZ 0.24802109227\n",
      "PLZ8_ANTG1 0.252138882878\n",
      "PLZ8_GBZ 0.252573610496\n",
      "KBA05_ANTG3 0.274722848522\n",
      "KBA05_ANTG4 0.275388982563\n",
      "KBA05_ANTG2 0.275671700666\n",
      "KBA05_ANTG1 0.277848414778\n",
      "MOBI_REGIO 0.278491551405\n",
      "KBA05_GBZ 0.278537873931\n"
     ]
    }
   ],
   "source": [
    "# Map weights for the first principal component to corresponding feature names\n",
    "# and then print the linked values, sorted by weight.\n",
    "# HINT: Try defining a function here or in a new cell that you can reuse in the\n",
    "# other cells.\n",
    "\n",
    "weights(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SEMIO_REL -0.301503764276\n",
      "FINANZ_SPARER -0.283861530307\n",
      "SEMIO_KULT -0.266386583159\n",
      "SEMIO_PFLICHT -0.262919767036\n",
      "PRAEGENDE_JUGENDJAHRE_DECADE -0.251952373478\n",
      "SEMIO_TRADV -0.250642464995\n",
      "SEMIO_FAM -0.237971791596\n",
      "FINANZ_UNAUFFAELLIGER -0.229225444757\n",
      "FINANZ_ANLEGER -0.202173250723\n",
      "SEMIO_MAT -0.198763407572\n",
      "SEMIO_RAT -0.181787271718\n",
      "SEMIO_SOZ -0.148672538222\n",
      "HH_EINKOMMEN_SCORE -0.0751479226716\n",
      "SEMIO_VERT -0.073675664384\n",
      "ONLINE_AFFINITAET -0.0354807567842\n",
      "ORTSGR_KLS9 -0.0304199387462\n",
      "EWDICHTE -0.027401126965\n",
      "HEALTH_TYP -0.0264069845708\n",
      "ANZ_HAUSHALTE_AKTIV -0.0256396543834\n",
      "RELAT_AB -0.0167017533825\n",
      "ARBEIT -0.0159809736548\n",
      "CAMEO_INTL_2015_WEALTH -0.0117667746994\n",
      "OST_WEST_KZ_O -0.0115280116138\n",
      "W_KEIT_KIND_HH -0.00172175421227\n",
      "SOHO_KZ -0.000264743847901\n",
      "ANZ_PERSONEN 0.00128532631933\n",
      "FINANZ_HAUSBAUER 0.0024446656927\n",
      "ANZ_TITEL 0.00745793268154\n",
      "PRAEGENDE_JUGENDJAHRE_MOVEMENT 0.00959532806069\n",
      "REGIOTYP 0.00999193623753\n",
      "OST_WEST_KZ_W 0.0115280116138\n",
      "KKK 0.0115354631913\n",
      "ANZ_HH_TITEL 0.0157144471081\n",
      "KBA13_ANZAHL_PKW 0.0159367889257\n",
      "PLZ8_ANTG3 0.0195766833221\n",
      "PLZ8_ANTG4 0.0212525916231\n",
      "PLZ8_ANTG2 0.0217648037601\n",
      "BALLRAUM 0.0234464217084\n",
      "PLZ8_HHZ 0.0241395561828\n",
      "MIN_GEBAEUDEJAHR 0.0243162356467\n",
      "KBA05_ANTG3 0.0276542707221\n",
      "KBA05_ANTG4 0.0293627225072\n",
      "KBA05_ANTG2 0.0294231611862\n",
      "PLZ8_GBZ 0.0305239718824\n",
      "PLZ8_ANTG1 0.0316103359084\n",
      "CAMEO_INTL_2015_LIFE_STAGE 0.0325590048554\n",
      "INNENSTADT 0.0326543066358\n",
      "GEBAEUDETYP_RASTER 0.0357130745255\n",
      "KBA05_GBZ 0.036256523208\n",
      "KBA05_ANTG1 0.0378755651521\n",
      "MOBI_REGIO 0.0384512900587\n",
      "KONSUMNAEHE 0.0415912771597\n",
      "GREEN_AVANTGARDE 0.0440321355918\n",
      "RETOURTYP_BK_S 0.0598080929255\n",
      "WOHNDAUER_2008 0.0791966580735\n",
      "SEMIO_DOM 0.0832865550721\n",
      "ALTERSKATEGORIE_GROB 0.109579762325\n",
      "SEMIO_KAEM 0.11545994899\n",
      "ANREDE_KZ 0.117574923768\n",
      "SEMIO_KRIT 0.13405527533\n",
      "FINANZ_MINIMALIST 0.174804308437\n",
      "SEMIO_LUST 0.189125728252\n",
      "FINANZ_VORSORGER 0.263142332695\n",
      "SEMIO_ERL 0.268298119271\n"
     ]
    }
   ],
   "source": [
    "# Map weights for the second principal component to corresponding feature names\n",
    "# and then print the linked values, sorted by weight.\n",
    "\n",
    "weights(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ANREDE_KZ -0.350271430137\n",
      "SEMIO_KAEM -0.338402301692\n",
      "SEMIO_DOM -0.304927770834\n",
      "SEMIO_KRIT -0.267627319331\n",
      "SEMIO_RAT -0.234155065759\n",
      "FINANZ_ANLEGER -0.187506260693\n",
      "FINANZ_SPARER -0.150587927543\n",
      "SEMIO_ERL -0.143812208183\n",
      "FINANZ_HAUSBAUER -0.131628381744\n",
      "HH_EINKOMMEN_SCORE -0.115927075039\n",
      "FINANZ_UNAUFFAELLIGER -0.114663577261\n",
      "SEMIO_PFLICHT -0.112952538745\n",
      "PRAEGENDE_JUGENDJAHRE_DECADE -0.0986463056765\n",
      "SEMIO_TRADV -0.0965713825255\n",
      "HEALTH_TYP -0.0701394839075\n",
      "ORTSGR_KLS9 -0.0541303055146\n",
      "EWDICHTE -0.0525509503482\n",
      "PRAEGENDE_JUGENDJAHRE_MOVEMENT -0.0472815507681\n",
      "ANZ_HAUSHALTE_AKTIV -0.0469919670817\n",
      "PLZ8_ANTG3 -0.0382474409184\n",
      "PLZ8_ANTG4 -0.0356527640898\n",
      "PLZ8_ANTG2 -0.0352469229924\n",
      "PLZ8_HHZ -0.0315984540711\n",
      "RELAT_AB -0.03128991523\n",
      "CAMEO_INTL_2015_WEALTH -0.0307348828823\n",
      "ARBEIT -0.0284153870513\n",
      "PLZ8_GBZ -0.0222310087556\n",
      "PLZ8_ANTG1 -0.0207398888474\n",
      "W_KEIT_KIND_HH -0.0184279605142\n",
      "OST_WEST_KZ_O -0.0136808022321\n",
      "KBA05_ANTG3 -0.0120053798001\n",
      "KBA05_ANTG4 -0.0102996575505\n",
      "KBA05_ANTG2 -0.0088372470638\n",
      "SOHO_KZ 0.00159425135409\n",
      "KBA05_GBZ 0.00244316094098\n",
      "MIN_GEBAEUDEJAHR 0.00289993287693\n",
      "KBA05_ANTG1 0.00343789742975\n",
      "MOBI_REGIO 0.00459222393371\n",
      "REGIOTYP 0.00486957145205\n",
      "KKK 0.00773493906575\n",
      "ANZ_TITEL 0.00887477999509\n",
      "ONLINE_AFFINITAET 0.0101498875896\n",
      "OST_WEST_KZ_W 0.0136808022321\n",
      "CAMEO_INTL_2015_LIFE_STAGE 0.0138549445938\n",
      "ANZ_HH_TITEL 0.0139986817954\n",
      "SEMIO_MAT 0.0197133301021\n",
      "SEMIO_REL 0.0245362957842\n",
      "KBA13_ANZAHL_PKW 0.0245766507013\n",
      "ALTERSKATEGORIE_GROB 0.0339352158777\n",
      "BALLRAUM 0.0381915542331\n",
      "GEBAEUDETYP_RASTER 0.0413875821064\n",
      "RETOURTYP_BK_S 0.0452305225444\n",
      "INNENSTADT 0.048370182269\n",
      "ANZ_PERSONEN 0.0489115372907\n",
      "KONSUMNAEHE 0.0594223108886\n",
      "WOHNDAUER_2008 0.0607658388104\n",
      "SEMIO_LUST 0.0727113280605\n",
      "GREEN_AVANTGARDE 0.0878255836868\n",
      "FINANZ_VORSORGER 0.129712366493\n",
      "SEMIO_KULT 0.200311731973\n",
      "SEMIO_FAM 0.203817223777\n",
      "SEMIO_SOZ 0.230318084204\n",
      "FINANZ_MINIMALIST 0.247350492214\n",
      "SEMIO_VERT 0.343575531196\n"
     ]
    }
   ],
   "source": [
    "# Map weights for the third principal component to corresponding feature names\n",
    "# and then print the linked values, sorted by weight.\n",
    "\n",
    "weights(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Discussion 2.3: Interpret Principal Components\n",
    "\n",
    "It could be observed that:\n",
    "- The first component is heavily (positively) influenced by KBA05_ANTG3 KBA05_ANTG4 KBA05_ANTG2 KBA05_ANTG1 MOBI_REGIO KBA05_GBZ KBA05_GBZ → Micro-cell features and movement patterns\n",
    "- The second component is heavily (negatively) influenced by SEMIO_REL FINANZ_SPARER SEMIO_KULT SEMIO_PFLICHT → Personality typology (religious or not, cultural-minded or not, dutiful or not) and Financial typology (money-saver or not)\n",
    "- The third component is heavily(positively) influenced by ANREDE_KZ SEMIO_KAEM SEMIO_DOM SEMIO_KRIT SEMIO_RAT → Gender and Personality typology (combative or not, dominant or not, critical or not, rational or not)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3: Clustering\n",
    "\n",
    "### Step 3.1: Apply Clustering to General Population\n",
    "\n",
    "You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.\n",
    "\n",
    "- Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.\n",
    "- Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.\n",
    "- Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.\n",
    "- Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_kmeans_score(data, center):\n",
    "    '''\n",
    "    returns the kmeans score regarding SSE for points to centers\n",
    "    INPUT:\n",
    "        data - the dataset you want to fit kmeans to\n",
    "        center - the number of centers you want (the k value)\n",
    "    OUTPUT:\n",
    "        score - the SSE score for the kmeans model fit to the data\n",
    "    '''\n",
    "    #instantiate kmeans\n",
    "    kmeans = KMeans(n_clusters=center)\n",
    "\n",
    "    # Then fit the model to your data using the fit method\n",
    "    model = kmeans.fit(data)\n",
    "    \n",
    "    # Obtain a score related to the model fit\n",
    "    score = np.abs(model.score(data))\n",
    "    \n",
    "    return score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Calculating for center #1\n",
      "Calculating for center #2\n",
      "Calculating for center #3\n",
      "Calculating for center #4\n",
      "Calculating for center #5\n",
      "Calculating for center #6\n",
      "Calculating for center #7\n",
      "Calculating for center #8\n",
      "Calculating for center #9\n",
      "Calculating for center #10\n",
      "Calculating for center #11\n",
      "Calculating for center #12\n",
      "Calculating for center #13\n",
      "Calculating for center #14\n",
      "Calculating for center #15\n",
      "Calculating for center #16\n",
      "Calculating for center #17\n",
      "Calculating for center #18\n",
      "Calculating for center #19\n",
      "Calculating for center #20\n",
      "Calculating for center #21\n",
      "Calculating for center #22\n",
      "Calculating for center #23\n",
      "Calculating for center #24\n",
      "Calculating for center #25\n",
      "Calculating for center #26\n",
      "Calculating for center #27\n",
      "Calculating for center #28\n",
      "Calculating for center #29\n",
      "Calculating for center #30\n"
     ]
    }
   ],
   "source": [
    "# Over a number of different cluster counts...\n",
    "    # run k-means clustering on the data and...\n",
    "    # compute the average within-cluster distances.\n",
    "    \n",
    "scores = []\n",
    "centers = list(range(1,31))\n",
    "\n",
    "for center in centers:\n",
    "    print(\"Calculating for center #{}\".format(center))\n",
    "    scores.append(get_kmeans_score(azdias_scaled_select, center))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f5efee71ac8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Investigate the change in within-cluster distance across number of clusters.\n",
    "# HINT: Use matplotlib's plot function to visualize this relationship.\n",
    "\n",
    "plt.plot(centers, scores, linestyle='--', marker='o', color='b');\n",
    "plt.xlabel('K');\n",
    "plt.ylabel('SSE');\n",
    "plt.title('SSE vs. K');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Re-fit the k-means model with the selected number of clusters and obtain\n",
    "# cluster predictions for the general population demographics data.\n",
    "\n",
    "# Using the elbow's method, 12 clusters are selected\n",
    "kmeans = KMeans(n_clusters=12)\n",
    "model = kmeans.fit(azdias_scaled_select)\n",
    "predictions_population = model.predict(azdias_scaled_select)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Discussion 3.1: Apply Clustering to General Population\n",
    "\n",
    "Based on the Score vs K graph, the number of clusters chosen was 12(elbow's method)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 3.2: Apply All Steps to the Customer Data\n",
    "\n",
    "Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.\n",
    "\n",
    "- Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.\n",
    "- Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)\n",
    "- Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load in the customer demographics data.\n",
    "customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv',sep=';')\n",
    "rows_customers = customers.shape[0]\n",
    "customers_clean = clean_data(customers)\n",
    "nan_rows_customers = rows_customers-customers_clean.shape[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Apply preprocessing, feature transformation, and clustering from the general\n",
    "# demographics onto the customer data, obtaining cluster predictions for the\n",
    "# customer demographics data.\n",
    "\n",
    "customers_imputed = pd.DataFrame(fill_NaN.transform(customers_clean))\n",
    "customers_imputed.columns = customers_clean.columns\n",
    "customers_imputed.index = customers_clean.index\n",
    "customers_scaled = scaler.transform(customers_imputed.values)\n",
    "customers_scaled_select = pca.transform(customers_scaled)\n",
    "predictions_customer = model.predict(customers_scaled_select)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 3.3: Compare Customer Data to Demographics Data\n",
    "\n",
    "At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.\n",
    "\n",
    "Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.\n",
    "\n",
    "Take a look at the following points in this step:\n",
    "\n",
    "- Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.\n",
    "  - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!\n",
    "- Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.\n",
    "- Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_cluster(clust_dict):\n",
    "    plt.figure(figsize=(20,6))\n",
    "    plt.bar(clust_dict.keys(),clust_dict.values(), color='g')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "def dict_per(dict):\n",
    "    sum = dict_sum(dict)\n",
    "    dict_return = dict.copy()\n",
    "    for e in dict_return.keys():\n",
    "        dict_return[e] = 100*dict_return[e]/sum\n",
    "    return dict_return"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "def dict_sum(dict):\n",
    "    sum = 0\n",
    "    for e in dict.values():\n",
    "        sum = sum + e\n",
    "    return sum"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{10: 129776, 6: 212670, 4: 38926, 11: 116242, 7: 108511, 3: 26112, 2: 9404, 9: 13226, 0: 105331, 5: 32427, 8: 591, 1: 4210, 12: 93795}\n",
      "{11: 65194, 6: 38119, 0: 17240, 3: 4034, 7: 2775, 4: 6462, 9: 614, 5: 1494, 10: 2858, 2: 884, 1: 566, 8: 31, 12: 51381}\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f5efee7fe10>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f5efee7f048>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Compare the proportion of data in each cluster for the customer data to the\n",
    "# proportion of data in each cluster for the general population.\n",
    "\n",
    "#Counting number of data points per cluster\n",
    "population_dict = dict(Counter(predictions_population))\n",
    "customer_dict = dict(Counter(predictions_customer))\n",
    "\n",
    "#Adding a cluster for the data points dropped in step 1.1.3\n",
    "population_dict.update({len(population_dict):len(nan_list_row)})\n",
    "customer_dict.update({len(population_dict)-1:nan_rows_customers})\n",
    "\n",
    "#Printing number of data points per cluster\n",
    "print(population_dict)\n",
    "print(customer_dict)\n",
    "\n",
    "#Percentage of data per cluster\n",
    "population_dict_per = dict_per(population_dict)\n",
    "customer_dict_per = dict_per(customer_dict)\n",
    "\n",
    "#Plotting percentage of data per cluster\n",
    "plot_cluster(population_dict_per)\n",
    "plot_cluster(customer_dict_per)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_center_dict(center_dict):\n",
    "    for e in center_dict:\n",
    "        print(\"{} {}\".format(e,center_dict[e]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_center_dict(cluster):\n",
    "    center = scaler.inverse_transform(pca.inverse_transform(model.cluster_centers_[cluster]))\n",
    "    center_dict = {}\n",
    "    i=0\n",
    "    for e in center:\n",
    "        center_dict.update({azdias.columns[i]:e})\n",
    "        i = i + 1\n",
    "    return center_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [],
   "source": [
    "# What kinds of people are part of a cluster that is overrepresented in the\n",
    "# customer data compared to the general population?\n",
    "\n",
    "# Calculate centroid for cluster #11 (this cluster has 35% representation in the \n",
    "# customer data but only 15% representation in the population data)\n",
    "center_dict_11 = create_center_dict(11)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "-\tALTERSKATEGORIE_GROB\t3.413249095\tage 30 - 45 years old/46 - 60 years old\n",
    "-\tANREDE_KZ\t1.056581671\tgender male\n",
    "-\tFINANZ_MINIMALIST\t4.636433264\tlow financial interest average/low\n",
    "-\tFINANZ_SPARER\t1.754140997\tmoney-saver very high/high\n",
    "-\tFINANZ_VORSORGER\t4.170584956\tbe prepared low\n",
    "-\tFINANZ_ANLEGER\t1.907603766\tinvestor high\n",
    "-\tFINANZ_UNAUFFAELLIGER\t1.995018278\tinconspicuous high\n",
    "-\tFINANZ_HAUSBAUER\t1.966159612\thome ownership high\n",
    "-\tGREEN_AVANTGARDE\t0.441111973\tnot a member of green avantgarde\n",
    "-\tHEALTH_TYP\t2.579539703\tsanitary affine / jaunty hedonists\n",
    "-\tRETOURTYP_BK_S\t4.330332001\tconservative Low-Returner\n",
    "-\tSEMIO_SOZ\t4.830835193\tsocially-minded average affinity/ low affinity\n",
    "-\tSEMIO_FAM\t4.543486461\tfamily-minded average affinity/low affinity\n",
    "-\tSEMIO_REL\t3.762080955\treligious high affinity/average affinity\n",
    "-\tSEMIO_MAT\t3.304032868\tmaterialistic high affinity\n",
    "-\tSEMIO_VERT\t5.970617137\tdreamful very low affinity\n",
    "-\tSEMIO_LUST\t5.10814603\tsensual-minded low affinity\n",
    "-\tSEMIO_ERL\t4.210504285\tevent-oriented average affinity\n",
    "-\tSEMIO_KULT\t5.05854446\tcultural-minded low affinity\n",
    "-\tSEMIO_RAT\t2.714429057\trational very high affinity/high affinity\n",
    "-\tSEMIO_KRIT\t3.409814246\tcritical-minded high affinity/average affinity\n",
    "-\tSEMIO_DOM\t3.499119944\tdominant-minded high affinity/average affinity\n",
    "-\tSEMIO_KAEM\t2.561188261\tcombative attitude very high affinity/high affinity\n",
    "-\tSEMIO_PFLICHT\t3.087840264\tdutiful high affinity\n",
    "-\tSEMIO_TRADV\t3.12127373\ttradional-minded high affinity\n",
    "-\tSOHO_KZ\t0.009558471\tno small office/home office\n",
    "-\tANZ_PERSONEN\t2.096473996\t2 adults in household\n",
    "-\tANZ_TITEL\t0.006468851\t0 professional academic title holders in household\n",
    "-\tHH_EINKOMMEN_SCORE\t3.246102844\thigh income\n",
    "-\tW_KEIT_KIND_HH\t5.56042298\tLikelihood of children in household unlikely/very unlikely\n",
    "-\tWOHNDAUER_2008\t8.463106632\tlength of residence 7-10 years/more than 10 years\n",
    "-\tANZ_HAUSHALTE_AKTIV\t2.803805217\t3 households in the building\n",
    "-\tANZ_HH_TITEL\t0.557264649\t0/1 professional academic title holders in building\n",
    "-\tKONSUMNAEHE\t3.806909535\tbuilding is located in a 1 x 1km grid cell that includes at least one RA1-consumption cell\n",
    "-\tMIN_GEBAEUDEJAHR\t1992.730517\tyear building was mentioned in the database\n",
    "-\tKBA05_ANTG1\t2.524173359\taverage share of 1-2 family homes/high share of 1-2 family homes\n",
    "-\tKBA05_ANTG2\t1.183942899\tlower share of 3-5 family homes\n",
    "-\tKBA05_ANTG3\t0.218280351\tno 6-10 family homes\n",
    "-\tKBA05_ANTG4\t0.064927471\tno 10+ family homes\n",
    "-\tKBA05_GBZ\t4.02047444\t17-22 buildings\n",
    "-\tBALLRAUM\t4.87482658\tDistance to nearest urban center 40 -  50 km\n",
    "-\tEWDICHTE\t3.077158397\t 90 - 149 households per km^2\n",
    "-\tINNENSTADT\t5.454112471\t10 - 20 km to city center/20 - 30 km to city center\n",
    "-\tGEBAEUDETYP_RASTER\t4.070528958\tmixed cell with low business share\n",
    "-\tKKK\t2.655129149\thigh/average\n",
    "-\tMOBI_REGIO\t4.04310608\tlow movement\n",
    "-\tONLINE_AFFINITAET\t3.346629365\thigh/very high\n",
    "-\tREGIOTYP\t4.06969839\tNeighborhood middle class\n",
    "-\tKBA13_ANZAHL_PKW\t686.3316102\t686 cars in the PLZ8 region\n",
    "-\tPLZ8_ANTG1\t2.839579213\taverage share of 1-2 family homes/high share of 1-2 family homes\n",
    "-\tPLZ8_ANTG2\t2.433457256\taverage share of 3-5 family homes/high share of 3-5 family homes\n",
    "-\tPLZ8_ANTG3\t1.001637776\tlower share of 6-10 family homes\n",
    "-\tPLZ8_ANTG4\t0.272881644\tno 10+ family homes\n",
    "-\tPLZ8_HHZ\t3.491535941\t300-599 households/600-849 households\n",
    "-\tPLZ8_GBZ\t3.854205405\t300-449 buildings/130-299 buildings\n",
    "-\tARBEIT\t2.752010121\tShare of unemployment in community average/low\n",
    "-\tORTSGR_KLS9\t4.109492169\tSize of community 10,001 to  20,000 inhabitants\n",
    "-\tRELAT_AB\t2.604052958\tShare of unemployment relative to county in which community is contained average/low\n",
    "-\tOST_WEST_KZ_O\t0.117534678\tNot from east germany\n",
    "-\tOST_WEST_KZ_W\t0.882465322\tfrom west germany\n",
    "-\tPRAEGENDE_JUGENDJAHRE_DECADE\t66.37129697\tDecade 60/70\n",
    "-\tPRAEGENDE_JUGENDJAHRE_MOVEMENT\t0.614420573\tMovement Avantgarde\n",
    "-\tCAMEO_INTL_2015_WEALTH\t2.597711018\tComfortable Households\n",
    "-\tCAMEO_INTL_2015_LIFE_STAGE\t3.595597601\tOlder Families &  Mature Couples/Families With School Age Children\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "# What kinds of people are part of a cluster that is underrepresented in the\n",
    "# customer data compared to the general population?\n",
    "\n",
    "# Calculate centroid for cluster #10(this cluster has 15% representation in the \n",
    "# population data but less than 5% representation in the customer data)\n",
    "center_dict_10 = create_center_dict(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "-\tALTERSKATEGORIE_GROB\t1.631263908\tage < 30 years old/30 - 45 years old\n",
    "-\tANREDE_KZ\t2.006875639\tgender female\n",
    "-\tFINANZ_MINIMALIST\t1.72653559\tlow financial interest very high/high\n",
    "-\tFINANZ_SPARER\t4.067882833\tmoney-saver low\n",
    "-\tFINANZ_VORSORGER\t2.199319085\tbe prepared high\n",
    "-\tFINANZ_ANLEGER\t4.141112044\tinvestor low\n",
    "-\tFINANZ_UNAUFFAELLIGER\t3.756023151\tinconspicuous average/low\n",
    "-\tFINANZ_HAUSBAUER\t3.587397327\thome ownership average/low\n",
    "-\tGREEN_AVANTGARDE\t0.127167866\tnot a member of green avantgarde\n",
    "-\tHEALTH_TYP\t2.214626179\tsanitary affine  \n",
    "-\tRETOURTYP_BK_S\t2.655915968\tdemanding Heavy-Returner/incentive-receptive Normal-Returner\n",
    "-\tSEMIO_SOZ\t2.970047132\tsocially-minded high affinity\n",
    "-\tSEMIO_FAM\t3.773792231\tfamily-minded high affinity/average affinity\n",
    "-\tSEMIO_REL\t5.362076132\treligious low affinity\n",
    "-\tSEMIO_MAT\t4.936684881\tmaterialistic low affinity\n",
    "-\tSEMIO_VERT\t2.286321171\tdreamful very high affinity\n",
    "-\tSEMIO_LUST\t2.827716111\tsensual-minded high affinity\n",
    "-\tSEMIO_ERL\t4.120311155\tevent-oriented average affinity\n",
    "-\tSEMIO_KULT\t4.214922044\tcultural-minded average affinity\n",
    "-\tSEMIO_RAT\t6.450145353\trational very low affinity/lowest affinity\n",
    "-\tSEMIO_KRIT\t5.457251907\tcritical-minded low affinity/very low affinity\n",
    "-\tSEMIO_DOM\t6.29692797\tdominant-minded very low affinity\n",
    "-\tSEMIO_KAEM\t5.877945948\tcombative attitude low affinity/very low affinity\n",
    "-\tSEMIO_PFLICHT\t6.173448268\tdutiful very low affinity\n",
    "-\tSEMIO_TRADV\t6.259928064\ttradional-minded very low affinity\n",
    "-\tSOHO_KZ\t0.008432122\tno small office/home office\n",
    "-\tANZ_PERSONEN\t1.708493147\t1/2 adults in household\n",
    "-\tANZ_TITEL\t0.002001289\t0 professional academic title holders in household\n",
    "-\tHH_EINKOMMEN_SCORE\t4.913337487\tlower income\n",
    "-\tW_KEIT_KIND_HH\t5.556542593\tLikelihood of children in household unlikely/very unlikely\n",
    "-\tWOHNDAUER_2008\t7.407000481\tlength of residence 6-7 years/7-10 years\n",
    "-\tANZ_HAUSHALTE_AKTIV\t9.425577505\t9/10 households in the building\n",
    "-\tANZ_HH_TITEL\t0.110827149\t0 professional academic title holders in building\n",
    "-\tKONSUMNAEHE\t2.848778544\tbuilding is located in a 500 x 500m grid cell that includes at least one RA1-consumption cell\n",
    "-\tMIN_GEBAEUDEJAHR\t1992.527475\tyear building was mentioned in the database\n",
    "-\tKBA05_ANTG1\t1.292729259\tlower share of 1-2 family homes/average share of 1-2 family homes\n",
    "-\tKBA05_ANTG2\t1.360049739\tlower share of 3-5 family homes/average share of 3-5 family homes\n",
    "-\tKBA05_ANTG3\t0.72125291\tno 6-10 family homes/lower share of 6-10 family homes\n",
    "-\tKBA05_ANTG4\t0.344672496\tno 10+ family homes/lower share of 10+ family homes\n",
    "-\tKBA05_GBZ\t2.998731112\t5-16 buildings\n",
    "-\tBALLRAUM\t3.992232072\tDistance to nearest urban center 30 -  40 km\n",
    "-\tEWDICHTE\t4.152715726\t150 - 319 households per km^2\n",
    "-\tINNENSTADT\t4.349498393\t5 - 10 km to city center/10 - 20 km to city center\n",
    "-\tGEBAEUDETYP_RASTER\t3.677129997\t mixed cell with middle business share/ mixed cell with low business share\n",
    "-\tKKK\t2.860338402\taverage/high\n",
    "-\tMOBI_REGIO\t2.7500006\tmiddle movement/high movement\n",
    "-\tONLINE_AFFINITAET\t3.448314298\thigh/very high\n",
    "-\tREGIOTYP\t4.59713041\tNeighborhood middle class/lower middle class\n",
    "-\tKBA13_ANZAHL_PKW\t601.7155165\t601 cars in the PLZ8 region\n",
    "-\tPLZ8_ANTG1\t2.14174601\taverage share of 1-2 family homes\n",
    "-\tPLZ8_ANTG2\t2.889117656\thigh share of 3-5 family homes/average share of 3-5 family homes\n",
    "-\tPLZ8_ANTG3\t1.715436476\taverage share of 6-10 family homes/lower share of 6-10 family homes\n",
    "-\tPLZ8_ANTG4\t0.778851724\tlower share of 10+ family homes/no 10+ family homes\n",
    "-\tPLZ8_HHZ\t3.643896708\t600-849 households/300-599 households\n",
    "-\tPLZ8_GBZ\t3.297044452\t130-299 buildings\n",
    "-\tARBEIT\t3.255039484\tShare of unemployment in community average\n",
    "-\tORTSGR_KLS9\t5.577375716\tSize of community 20,001 to  50,000 inhabitants/50,001 to 100,000 inhabitants\n",
    "-\tRELAT_AB\t3.157032947\tShare of unemployment relative to county in which community is contained average\n",
    "-\tOST_WEST_KZ_O\t0.215990617\tNot from east germany\n",
    "-\tOST_WEST_KZ_W\t0.784009383\tfrom west germany\n",
    "-\tPRAEGENDE_JUGENDJAHRE_DECADE\t86.08376904\tDecade 80/90\n",
    "-\tPRAEGENDE_JUGENDJAHRE_MOVEMENT\t0.139258233\tMovement Mainstream\n",
    "-\tCAMEO_INTL_2015_WEALTH\t3.700968134\tComfortable Households/Less Affluent Households\n",
    "-\tCAMEO_INTL_2015_LIFE_STAGE\t2.847298771\tFamilies With School Age Children/Young Couples With Children\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Discussion 3.3: Compare Customer Data to Demographics Data\n",
    "\n",
    "Based on the clustering analysis, we can say that:\n",
    "- Relatively popular:\n",
    "We can see that this population is represented by cluster #11. This cluster represents people(males) born in the 60's 70's(from 30 to 60 years old), originally from west germany. Wealth: comfortable households, life stage: older families & mature couples. They own their own house and have interest in finance and investing. They are conservative/traditional and dreamful affinity is low.\n",
    "- Relatively unpopular:\n",
    "We can see that this population is represented by cluster #10. This cluster represents people(females) born in the 80's 90's(less than 45 years old), originally from west germany. Wealth: less affluent households, life stage: families with achool age children. They don't own their own house and don't have interest in finance/investing. They are not religious/traditional and dreamful affinity is very-high. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
