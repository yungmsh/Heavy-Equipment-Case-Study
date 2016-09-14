from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

class ImputeValues(TransformerMixin):
    def fit(self, X, y):
        # Combine variations of EROPS AC
        return self

    def transform(self, X):
        X['Enclosure'] = X['Enclosure'].apply(lambda x: 'EROPS w AC' if x=='EROPS AC' else x)

        value_counts = X['Enclosure'].value_counts()
        enclosure_defaults = ['OROPS', 'EROPS', 'EROPS w AC']
        enclosure_others = set(value_counts.index) - set(enclosure_defaults)
        X['Enclosure'] = X['Enclosure'].apply(lambda x: X['Enclosure'].mode()[0] if x in enclosure_others else x)
        return X

class KeepDenseColumns(TransformerMixin):
    def fit(self, X, y):
        column_counts = X.apply(lambda x: x.count(), axis=0)
        self.keep_columns = column_counts[column_counts == column_counts.max()]
        return self

    def transform(self, X):
        return X.ix[:,self.keep_columns.index]

class BinaryColumns(TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X['datasource_132'] = X['datasource'].apply(lambda x: True if x==132 else False)
        return X

class ReplaceOutliers(TransformerMixin):
    def fit(self, X, y):
        self.replace_value = X.YearMade[X.YearMade > 1900].mode()[0]
        return self

    def transform(self, X):
        X['YearMade'] = X['YearMade'].apply(lambda x: self.replace_value if x==1000 else x)
        return X

class ComputeAge(TransformerMixin):
    '''Compute the age of the vehicle at sale, and set negative values to be the positive mean.
    '''
    def fit(self, X, y):
        return self

    def transform(self, X):
        X['Age'] = X['saledate'].dt.year - X.YearMade
        mean_age = X[X['Age']>=0]['Age'].mean()
        X['Age'] = X['Age'].apply(lambda x: round(mean_age) if x<0 else x)
        return X

class TemporalFeatures(TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X['sale_month'] = X['saledate'].dt.month
        X['sale_dayofmonth'] = X['saledate'].dt.day
        X['sale_dayofweek'] = X['saledate'].dt.dayofweek
        return X

class GeographicalFeatures(TransformerMixin):
    def fit(self, X, y):
        newengland = 'Connecticut, Maine, Massachusetts, New Hampshire, Rhode Island, Vermont'
        midatlantic = 'Delaware, Maryland, New Jersey, New York, Pennsylvania, Washington DC'
        south = 'Alabama, Arkansas, Florida, Georgia, Kentucky, Louisiana, Mississippi, Missouri, North Carolina, South Carolina, Tennessee, Virginia, West Virginia'
        midwest = 'Illinois, Indiana, Iowa, Kansas, Michigan, Minnesota, Nebraska, North Dakota, Ohio, South Dakota, Wisconsin'
        southwest = 'Arizona, New Mexico, Oklahoma, Texas'
        west = 'Alaska, California, Colorado, Hawaii, Idaho, Montana, Nevada, Oregon, Utah, Washington, Wyoming'
        regions = [newengland, midatlantic, south, midwest, southwest, west]
        self.region_names = ['newengland','midatlantic','south','midwest','southwest','west']
        self.regions_dict = {k: None for k in self.region_names}
        for region,lst in zip(self.region_names, regions):
            self.regions_dict[region] = lst.split(', ')

        return self

    def transform(self, X):
        self.X = X
        self.X['region'] = self.X['state'].apply(self.binStates)

        # Move uncategorized regions to largest bin
        # value_counts = self.X['region'].value_counts()
        # region_others = set(value_counts.index) - set(self.region_names)
        self.X['region'] = self.X['region'].apply(self.setSimilarRegion)
        return self.X

    def binStates(self, state):
        for name,lst in self.regions_dict.iteritems():
            if state in lst:
                return name
        return state

    def setSimilarRegion(self, region):
        if region not in self.region_names:
            # pivoted = self.X.pivot_table('SalePrice', 'region').sort_values(ascending=False)
            # difference = pivoted - self.X[self.X['region']==region]['SalePrice'].mean()
            # most_similar = np.argmin(abs(difference))
            # return most_similar
            if region == 'Puerto Rico':
                return 'midatlantic'
            else:
                return 'south'
        else:
            return region

class DummifyCategoricals(TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        output = pd.get_dummies(X, columns=['ProductGroup','region','Enclosure'])
        return output

class FinalColumns(TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        columns = ['Age', 'datasource_132', 'ProductGroup_BL', 'ProductGroup_MG',
       'ProductGroup_SSL', 'ProductGroup_TEX', 'ProductGroup_TTT',
       'ProductGroup_WL', 'region_midatlantic', 'region_midwest',
       'region_newengland', 'region_south', 'region_southwest',
       'region_west', 'sale_month', 'sale_dayofmonth', 'sale_dayofweek',
       'Enclosure_EROPS', 'Enclosure_EROPS w AC', 'Enclosure_OROPS']
        self.X = X[columns]
        return self.X

if __name__=='__main__':
    df = pd.read_csv('data/Train.csv', parse_dates=['saledate'])
    y = df.SalePrice

    pipeline = Pipeline([
        ('impute', ImputeValues()),
        # ('dense', KeepDenseColumns()),
        ('binary', BinaryColumns()),
        ('outliers', ReplaceOutliers()),
        ('age', ComputeAge()),
        ('temporal', TemporalFeatures()),
        ('geographical', GeographicalFeatures()),
        ('dummify', DummifyCategoricals()),
        ('final', FinalColumns()),
        ('rf', RandomForestRegressor(n_estimators=100, min_samples_split=20, min_samples_leaf=10))
    ])

    def rmsle(y_hat, y):
        target=y
        predictions = y_hat
        log_diff = np.log(predictions+1) - np.log(target+1)
        return np.sqrt(np.mean(log_diff**2))

    #GridSearch
    # params = {'nearest_average__window':[3,5,7]}

    #Turns our rmsle func into a scorer of the type required
    #by gridsearchcv.
    # rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

    # gscv = GridSearchCV(p, params,
    #                     scoring=rmsle_scorer,
    #                     cv=cross_val)
    # clf = gscv.fit(df.reset_index(), y)

    # print 'Best parameters: %s' % clf.best_params_
    # print 'Best RMSLE: %s' % clf.best_score_

    test = pd.read_csv('data/test.csv', parse_dates=['saledate'])
    test = test.sort_values(by='SalesID')

    model = pipeline.fit(df, y)
    test_predictions = model.predict(test)

    soln = pd.read_csv('data/do_not_open/test_soln.csv')
    test_actual = soln.SalePrice

    print 'RMSLE score:', rmsle(test_predictions, test_actual)
