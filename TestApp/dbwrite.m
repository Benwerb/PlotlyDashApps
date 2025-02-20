% Define database connection parameters
dbname = 'SprayData';
username = 'spraydabase_user';
password = '8pgy9Sba79ETgds8QcaycQj0U6uIhhwQ';
host = 'dpg-cur2o7lds78s7384jthg-a'; % This could be a DSN (Data Source Name)
port = 5432;

jdbcDriver = 'org.postgresql.Driver';
jdbcURL = ['jdbc:postgresql://' host ':' port '/' dbName];

% Create database connection
conn = database(dbName, username, password, jdbcDriver, jdbcURL);

% Check if connection was successful
if isopen(conn)
    disp('Connection successful!');
else
    disp('Connection failed!');
    disp(conn.Message);
end
