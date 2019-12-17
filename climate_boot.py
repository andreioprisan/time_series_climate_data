"""
    Climate Data Circular Bootstrap
    
    Purpose: executes nonparametric block bootstrap (Supplementary Section Method IC)
    on either GISTEMP or HADCRUT temperature anomaly data
    
    Usage: set current directory to folder containing script, then enter following as 1 line in the command line:
    
    python3 climate_boot.py --time_freq month --start 1/1998 --end 12/2013 --source gistemp --block 6 --file xlsx
    
    
    Note:
    an Excel or csv file will be generated in the working directory named based on command line inputs
    contains bootstrap standard errors for beta_1_hat at various bootstap block sizes and p-values
    
    
"""
import click
import requests
import io
import numpy as np
import pandas as pd
import statsmodels.api as sm
import random
from random import choices
from scipy.stats import t, norm


class ClimateDataCircularBootstrap(object):
    '''class object to retrive climate data and conduct Method 1C'''
    def __init__(self, time_freq, start, end, source, block, file):
        """Initializes ClimateDataCircularBootstrap object's member variables

        :param str time_freq: frequency level (year or month)
        :param str start: start time of bootstrap (year or month/year)
        :param str end: end time of bootstrap (year or month/year)
        :param str source: database to fetch climate data from (hadcrut or gistemp)
        :param int block: max block size to conduct bootstrap analysis on
        :param str file: output results table file type (csv or xlsx)
        """
        self.time_freq = time_freq
        if time_freq == 'month':
            s_mn, s_yr = str.split(start, sep='/')
            e_mn, e_yr = str.split(end, sep='/')
            self.start_str = s_mn + '-' + s_yr
            self.end_str = e_mn + '-' + e_yr
            self.start_num = int(s_yr) + ((int(s_mn) - 1) / 12)
            self.end_num = int(e_yr) + ((int(e_mn) - 1) / 12)
        else:
            self.start_num = int(start)
            self.end_num = int(end)
            self.start_str = start
            self.end_str = end
        self.source = source
        self.block = block
        self.file = file

    def read_hd(self):
        """Reads in hadcrut data according to frequency of interest

        :return: hadcrut data
        :rtype: DataFrame
        """
        url_freq = 'annual' if self.time_freq == 'year' else 'monthly'
        url = "https://www.metoffice.gov.uk/hadobs/hadcrut4/data/current/time_series/HadCRUT.4.6.0.0.%s_ns_avg.txt" % (
            url_freq)
        response = requests.get(url)
        file_object = io.StringIO(response.content.decode('utf-8'))

        return pd.read_csv(file_object, sep="   ", header=None, engine='python')

    def get_boot_data(self):
        """Retrieves and modifies climate data in a standardized format for upcoming bootstrap algorithm

        :return: data of interest of two columns: time and (temperature) change
        observations within start and end timeframe
        :rtype: DataFrame
        """
        ds = self.read_hd() if self.source == 'hadcrut' else pd.read_csv(
            'https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv').reset_index()

        def make_na(cell):
            '''change *** placeholder cells to nan'''
            if cell == '***':
                return np.nan

            return cell

        def had_m(ds):
            '''restructure hadcrut monthly climate data'''
            ds['avg'] = ds.iloc[:, 1:].apply(np.mean, axis=1)
            ds['time'] = [int(str.split(ds.iloc[0, 0], '/')[0]) + i * 1 / 12 for i in range(ds.shape[0])]

            return ds[['time', 'avg']]

        def had_y(ds):
            '''restructure hadcrut annual climate data'''
            ds['avg'] = ds.iloc[:, 1:].apply(np.mean, axis=1)

            return ds.iloc[:, [0, -1]]

        def gis_m(ds):
            '''restructure gistemp monthly climate data'''
            ds = ds.rename(columns=ds.iloc[0]).iloc[1:, :13].applymap(lambda x: float(make_na(x)))
            new_ds = pd.Series([ds.iloc[0, 0] + i * 1 / 12 for i in range(ds.shape[0] * 12)]).to_frame()
            new_ds['avg'] = pd.concat([ds.iloc[i, 1:13] for i in range(ds.shape[0])], axis=0).reset_index(drop=True)

            return new_ds

        def gis_y(ds):
            '''restructure gistemp annual climate data'''
            ds = ds.rename(columns=ds.iloc[0]).iloc[1:, :13].applymap(lambda x: float(make_na(x)))
            ds['avg'] = ds.iloc[:, 1:].apply(np.mean, axis=1)
            return ds.iloc[:, [0, -1]]

        '''dictionary of four functions above'''
        modify_df = {'hadcrut_month': had_m, 'hadcrut_year': had_y, 'gistemp_month': gis_m, 'gistemp_year': gis_y}

        final_df = modify_df[self.source + '_' + self.time_freq](ds)
        final_df.columns = ['time', 'change']
        epsilon = 10e-5

        return final_df[(final_df['time'] >= self.start_num - epsilon) & (final_df['time'] <= self.end_num + epsilon)]

    def get_blocks_list(self, resid_df, b):
        """Generates a list of DataFrames, each being a block to be considered in circular bootstrap

        :param DataFrame resid_df: residuals after fitting OLS to climate data
        :param int b: size of blocks to generate
        :return: list of block DataFrames
        :rtype: list
        """
        # largest block size -> 1 block total
        if b == resid_df.shape[0]:
            return [resid_df]
        # get df of each block
        block_df = pd.concat([resid_df, resid_df.iloc[:(b - 1), :]], axis=0)
        # return list of df blocks
        return [block_df.iloc[i:(i + b), :] for i in range(resid_df.shape[0])]

    def get_bootstrap_sample(self, resid_df, predictions, block_size, seed):
        """Randomly selects blocks and calculates final bootstrap sample

        :param DataFrame resid_df: residuals after fitting original OLS
        :param DataFrame predictions: predicted values of original OLS
        :param int block_size: size of block to do bootstrap
        :param int seed: seed number (dictates block selections)
        :return: new "stationary" climate time series
        :rtype: DataFrame
        """
        random.seed(seed)
        block_df = self.get_blocks_list(resid_df, block_size)
        innov = choices(block_df, k=round(resid_df.shape[0] / block_size) + 1)
        innov_df = pd.concat(innov).reset_index(drop=True).iloc[:resid_df.shape[0], :]

        return predictions + innov_df

    def get_block_results_row(self, resid_df, X, predictions, block_size):
        """Execute operations in Method IC to yield results (standard error & p-values) bootstrap of specified size

        :param DataFrame resid_df: residuals from original OLS
        :param DataFrame X: actual time series temperature values
        :param DataFrame predictions: predicted values from original OLS
        :param int block_size: block size to execute circular bootstrap for
        :return: series representing a row of the results table
        :rtype: Series
        """
        Y = [self.get_bootstrap_sample(resid_df, predictions, block_size, i) for i in range(1000)]

        regrs = [sm.OLS(boot_sample, X).fit() for boot_sample in Y]
        beta_1s = [regr_model.params['time'] for regr_model in regrs]
        std_err = pd.Series([regr_model.HC0_se['time'] for regr_model in regrs]).mean()

        B = 1000
        n = X.shape[0] - 2
        var_b1 = (1 / (B - 1)) * sum([(beta_1 - ((1 / B) * sum(beta_1s))) ** 2 for beta_1 in beta_1s])
        wald_stat = pd.Series(beta_1s).mean() / (var_b1 ** (1 / 2))

        pval_t = pd.Series(2 * t.cdf(-wald_stat, n)).mean()
        pval_z = pd.Series(2 * norm.cdf(-wald_stat)).mean()
        pval_boot = (1 / B) * pd.Series(
            abs(pd.Series(beta_1s) - pd.Series(beta_1s).mean()) > pd.Series(beta_1s).mean()).apply(int).sum()

        return pd.Series([block_size, std_err, pval_t, pval_z, pval_boot])

    def get_block_results_all(self, resid_df, X, predictions):
        """Iterates get_block_results_row to get entire results table of all block sizes of interest

        :param DataFrame resid_df: residuals from original OLS
        :param DataFrame X: actual time series temperature values
        :param DataFrame predictions: predicted values from original OLS
        :return: results dataframe for all block sizes of interest (block sizes 1 to self.block)
        :rtype: DataFrame
        """
        results_df = pd.concat([self.get_block_results_row(resid_df, X, predictions, i + 1) for i in range(self.block)],
                               axis=1).transpose()
        results_df.columns = ['Block Size', 'Standard Error', 't%s' % str(resid_df.shape[0] - 2), 'N(0,1)', 'Bootstrap']

        return results_df

    def execute_bootstrap(self):
        """Carry out entire algorithm, add OLS results to table, and create/save the results to specified file type

        """
        df = self.get_boot_data()
        X = df['time'].reset_index(drop=True)
        y = df['change'].reset_index(drop=True)
        X = sm.add_constant(X)
        reg = sm.OLS(y, X).fit()

        predictions = pd.Series(reg.predict(X)).to_frame().reset_index(drop=True)
        resid_df = pd.Series(reg.predict(X) - y).to_frame().reset_index(drop=True)
        final_df = self.get_block_results_all(resid_df, X, predictions)
        no_boot = pd.DataFrame(columns=final_df.columns)
        no_boot.loc[0] = [np.nan, reg.HC0_se['time'], reg.pvalues['time'],
                          2 * norm.cdf(-reg.summary2().tables[1]['t']['time']), np.nan]
        final_df = final_df.append(no_boot, ignore_index=True)

        output = '%s_to_%s_%s_%s_b%s_circ_block_boot_results_table.%s' % (
            self.start_str, self.end_str, self.source, self.time_freq, self.block, self.file)

        if self.file == 'xlsx':
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            final_df.to_excel(writer, index=None, sheet_name='results')
            writer.save()

        else:
            final_df.to_csv(output)


@click.command()
@click.option('--time_freq', type=str, help='Time frequency: year or month')
@click.option('--start', type=str, help='Start year (2018) or month and year (4/2018) of interest')
@click.option('--end', type=str, help='End year (2019) or month and year (4/2019) of interest')
@click.option('--source', type=str, help='Data of interest: hadcrut or gistemp')
@click.option('--block', type=int, help='Maximum block size to conduct circular bootstrap on')
@click.option('--file', type=str, help='Results table output type: csv or xlsx')

def conduct_bootstrap_analysis(time_freq, start, end, source, block, file):
    """High level function executed: create bootstrap object and run algorithm

    :param str time_freq: frequency level (year or month)
    :param str start: start time of bootstrap (year or month/year)
    :param str end: end time of bootstrap (year or month/year)
    :param str source: database to fetch climate data from (hadcrut or gistemp)
    :param int block: max block size to conduct bootstrap analysis on
    :param str file: output results table file type (csv or xlsx):
    """
    cdcb = ClimateDataCircularBootstrap(time_freq, start, end, source, block, file)
    cdcb.execute_bootstrap()


if __name__ == '__main__':
    conduct_bootstrap_analysis()
