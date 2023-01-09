import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

class data_prepare_pipeline:
    def __init__(self, data, timezone_map_dict, starting_event, end_event, informative_landing_pages):
        self.starting_event = starting_event
        self.end_event = end_event
        self.data = data
        self.timezone_map_dict = timezone_map_dict
        self.session_ids = list(data['SESSION_ID'].unique())
        self.informative_landing_pages = informative_landing_pages
        
    def _map_to_local_time(self, row):
        if row['REGION'] in self.timezone_map_dict:
            td = self.timezone_map_dict[row['REGION']]
        else:
            td = -5
        if td >= 0:
            return row['EVENT_CREATED_AT'] + timedelta(hours=td)
        else:
            td = -1 * td
            return row['EVENT_CREATED_AT'] - timedelta(hours=td)

        

    def _feature_engineering(self):
        # parse timestamp
        self.data['EVENT_CREATED_AT'] = self.data['EVENT_CREATED_AT'].apply(lambda e:datetime.strptime(e,'%Y-%m-%d %H:%M:%S.%f'))
        # parse joined date
        self.data['JOINDATE'] = self.data['JOINDATE'].apply(lambda e:datetime.strptime(e,'%Y-%m-%d %H:%M:%S.%f'))

        # change timezone
        self.data['EVENT_CREATED_AT'] = self.data.apply(self._map_to_local_time, axis = 1)

    def _long_table_new_row_intialization(self, session_data):

        # initialze the final prediction table row 
        long_table_new_row = {}

        # initialize identifier: used for train test split in boosted tree training
        long_table_new_row['user_id'] = session_data['USER_ID'][session_data.index[0]]
        long_table_new_row['session_id'] = session_data['SESSION_ID'][session_data.index[0]]

        # initialize accumulate time
        long_table_new_row['accumulate_time'] = 0

        # initialize shared features
        long_table_new_row['year'] = session_data['EVENT_CREATED_AT'][session_data.index[0]].year
        long_table_new_row['month'] = session_data['EVENT_CREATED_AT'][session_data.index[0]].month
        long_table_new_row['hour'] = session_data['EVENT_CREATED_AT'][session_data.index[0]].hour
        long_table_new_row['weekday'] = session_data['EVENT_CREATED_AT'][session_data.index[0]].weekday()

        long_table_new_row['old_customer'] = int(session_data['JOINDATE'][session_data.index[0]] < datetime.now() + relativedelta(months=-1))
        # long_table_new_row['device'] = session_data['DEVICE_TYPE'][session_data.index[0]]

        if session_data['LANDING_PAGE'][session_data.index[0]] in self.informative_landing_pages:
            long_table_new_row['landing_page'] = session_data['LANDING_PAGE'][session_data.index[0]]
        else:
            long_table_new_row['landing_page'] = 'Others'

        long_table_new_row['platform_generalized'] = session_data['PLATFORM'][session_data.index[0]].split(' ')[0]
        long_table_new_row['covid'] = int(session_data['EVENT_CREATED_AT'][session_data.index[0]] > datetime(2020,3,1))

        # initialize response
        long_table_new_row['reached'] = int(any([e for e in session_data['EVENT_NAME'] if e == self.end_event]))

        return long_table_new_row


    def _long_table_update_row(self, event_header, long_table_new_row, row, addTime = True):
        """update long_table_new_row with new event happened
        
        Args:
            event_header(str): the name of the event
            long_table_new_row (dict): dictionary of the row
            row (:obj: `pandas.Series`): current row of the pandas dataframe
            addTime: whether change event time. Default = True.
        
        Return:
            long_table_new_row (dict): dictionary of the row with updated event & time
        """
        if event_header in long_table_new_row:
            long_table_new_row[event_header] += 1
            if(addTime):
                long_table_new_row[event_header+'_time'] += row['event_time_consumed']
        else:
            long_table_new_row[event_header] = 1
            if(addTime):
                long_table_new_row[event_header+'_time'] = row['event_time_consumed']
        return long_table_new_row

    def _one_hot_encoder(self, data):
        categorical_cols = data.select_dtypes(include = ['bool','object']).columns
        data = pd.get_dummies(data, columns = categorical_cols)
        return data

    def get_long_table_rows(self, session_id):
        # get session data subset
        session_data = self.data[self.data['SESSION_ID'] == session_id]
        
        # sort session data by time
        session_data = session_data.sort_values(by = "EVENT_CREATED_AT")

        # get rid of events happend before visit pq page:
        # session_data = session_data.sort_values(by = "EVENT_CREATED_AT")
        # session_data = session_data.iloc[session_data[session_data['EVENT_NAME'] == self.starting_event].index[0]:,:]

        # calculate event time consumed, last event will have time_consumed == -1
        session_data['event_time_consumed'] = [e.total_seconds() for e in list(session_data['EVENT_CREATED_AT'].diff(periods=1)[1:])] + [-1]

        # initialze the final prediction table row
        long_table_new_row = self._long_table_new_row_intialization(session_data)

        # list to store all rows
        session_long_data_rows = []

        # generate long table for session
        for i,row in session_data.iterrows():
            # if reached to target path then break
            if row['EVENT_NAME'] == self.end_event:
                break

            # if last row: only event count +1 
            if row['event_time_consumed'] == -1:
                event_header = row['EVENT_NAME']
                long_table_new_row = self._long_table_update_row(event_header, long_table_new_row, row, addTime = False)               
      
                #record the row
                row_to_append = long_table_new_row.copy()
                session_long_data_rows.append(row_to_append)
                break 

            #if it's a regular event row, count event +1 and time + time consumed
            event_header = row['EVENT_NAME']
            long_table_new_row = self._long_table_update_row(event_header, long_table_new_row, row)
       
            #update accumulate time as well
            long_table_new_row['accumulate_time'] += row['event_time_consumed']

            #record the row
            row_to_append = long_table_new_row.copy()
            session_long_data_rows.append(row_to_append)
        
        return session_long_data_rows


    
    def prepare_data(self):
        # feature engineering
        self._feature_engineering()

        # list to hold all long data rows
        long_table_rows = []

        # iterate session_ids to get long table for each session
        for i,session_id in enumerate(self.session_ids):
            # progress tracking
            if i % 10000 == 0:
                print('{0}/{1} Session Processed Check Point.'.format(i,len(self.session_ids)))

            # get long format table for session
            session_long_table_rows = self.get_long_table_rows(session_id) 

            # append tables to output tables
            long_table_rows += session_long_table_rows
        
        output_long_table = pd.DataFrame.from_dict(long_table_rows).reset_index(drop = True)

        # one hot encode categorical variables
        output_long_table = self._one_hot_encoder(output_long_table)

        return output_long_table
