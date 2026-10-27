from multiprocessing import cpu_count, get_context

#import deduce
import benedict
import numpy as np
import pandas as pd
import regex as re
from sklearn.base import BaseEstimator, TransformerMixin

single_letter_spaces_re = re.compile(r'(?<=(?<!\\pL)\\pL) (?=\\pL(?!\\pL))')
splitter_re = re.compile(r'[^\w]', re.UNICODE)
def read_in_config(file):
    return benedict.benedict(file)

class Cleaner(BaseEstimator, TransformerMixin):
    '''Cleaner class

        - removal of overhead, exotic character groups, text-decoding errors, etc.
        - minimal word count
    '''
    def __init__(self, replacements=[(re.compile(r'[^\w]{2,}'), ' ')],
                 stopwords=[], lower=False, n_jobs=-1, to_dataframe=False,
                 data_index=None, text_cols=None):
        self.replacements = replacements
        self.stopwords = stopwords
        self.lower = lower
        self.n_jobs = n_jobs
        self.to_dataframe = to_dataframe
        self.data_index = data_index
        self.text_cols = text_cols
        assert isinstance(stopwords, list), "stopwords is not a list.."

    def remove_stopwords(self, wlist):
        res = []
        for idx, _w in enumerate(wlist):
            if _w not in self.stopwords:
                res.append(_w)
        return " ".join(res)

    def _to_dataframe(self, X):
        '''
            Assume that X is a list of lists with text -> pd.DataFrame
        '''
        output = pd.DataFrame()
        if self.data_index is None:
            self.data_index = list(range(len(X[0])))
        if self.text_cols is None:
            self.text_cols = [f'column_{i}' for i in range(len(X))]
        for idx, c in enumerate(X):
            _X = np.array(c)
            temp = pd.DataFrame(
                         data=_X,
                         columns=[self.text_cols[idx]],
                         index=self.data_index
                         )
            output = pd.concat([output, temp], axis=1)
        return output

    def _clean_text_series(self, txt):
        '''
            txt : list of strings
        '''
        #print("Processing replacements..., txt type {}".format(type(txt)))
        txt = pd.Series(txt)
        try:
            for _regtup in self.replacements:
                txt = txt.str.replace(_regtup[0], _regtup[1], regex=True)
        except AttributeError as e:
            msg = "Error with tuple {},\n\t txt.type {}\n\t \
            e: {}, txt: {}".format(_regtup, type(txt), e, txt.values[:1])
            raise Exception('AttributeError', msg)

        if self.lower:
            txt = txt.str.lower()
        if (isinstance(self.stopwords, list)) & (len(self.stopwords)>0):
            stopword_replace = \
                lambda x: self.remove_stopwords(splitter_re.split(x))
            txt = txt.apply(stopword_replace)
        return list(txt)

    def _clean_text_list(self, txt):
        '''
            txt : list of strings
        '''
        #print(f"Processing replacements..., txt type {type(txt)}, len {len(txt)}")
        num_rows = len(txt)
        cleaned = np.array(txt, dtype=str)
        for _regtup in self.replacements:
            for idx, _str in enumerate(cleaned):
                new = _regtup[0].sub(_regtup[1], _str)
                cleaned[idx] = new

        if self.lower:
            cleaned = np.char.lower(cleaned)

        if (isinstance(self.stopwords, list)) & (len(self.stopwords)>0):
            for idx in range(num_rows):
                cleaned[idx] = self.remove_stopwords(splitter_re.split(cleaned[idx]))
        return list(cleaned)

    def _par_clean_text(self, txts):
        ''' Apply a function separately to each column in a dataframe, in parallel.'''
        if self.n_jobs==-1:
            self.n_jobs = cpu_count()

        with get_context("spawn").Pool(self.n_jobs) as pool:
            seq = txts
            chunk = int(len(seq)/self.n_jobs)
            stuple = tuple([seq[i*chunk:(i+1)*chunk]
                        for i in range(0, self.n_jobs)])
            # add remainder to tuple
            stuple += (seq[self.n_jobs*chunk:],)
            #results_list = pool.map(self._clean_text_series, stuple)
            results_list = pool.map(self._clean_text_list, stuple)

        return [_res for res in results_list for _res in res]

    def _data_check(self, X):
        err_str = 'X should be a pandas.Series, a pandas.DataFrame, ' \
                    'a np.array or a list of strings'
        if isinstance(X, list):
            # check if all items are lists, in which all items are text
            # of if all items is text
            all_list = True
            all_text_1 = True
            all_text_2 = True

            for el in X:
                all_list &= isinstance(el, list)
                all_text_1 &= isinstance(el, str)
                if all_list:
                    for _el in el:
                        all_text_2 &= isinstance(_el, str)
            if all_list & all_text_2:
                return X
            elif all_text_1:
                return [X]
            else:
                raise ValueError(err_str)
        elif isinstance(X, pd.DataFrame):
            self.data_index = X.index
            assert all([dt is object for dt in X.dtypes]), \
                'X should be pd.DataFrame with only object dtype'
            self.text_cols = X.columns
            return X.T.values.tolist()
        elif isinstance(X, pd.Series):
            assert X.dtype==object, 'X should be pd.Series with object dtype'
            return [X.T.values.tolist()]
        elif isinstance(X, np.ndarray):
            assert (X.dtype == np.dtype('object'))|(X.dtype==np.dtype('<U1')), \
                'X should be np.array of strings'
            if len(X.shape)>1:
                return X.T.tolist()
            else:
                return [X.T.tolist()]

    def fit(self, X, y=None):
        assert (isinstance(X, list) |
                isinstance(X, pd.DataFrame) |
                isinstance(X, pd.Series) |
                isinstance(X, np.ndarray)), "X should be lists of lists, \
                a list of text, pd.DataFrame or np.array"
        _X = self._data_check(X)
        self.cleaned = []
        for _col in _X:
            if self.n_jobs==1:
                self.cleaned.append(self._clean_text_list(_col))
            else:
                self.cleaned.append(self._par_clean_text(_col))
        return self

    def transform(self, X, y=None):
        if self.to_dataframe:
            return self._to_dataframe(self.cleaned)
        else:
            return self.cleaned

