--  Script "Create_userid.sql"
--  Create the variable &userid that will contain the Oracle userid’s name as FirstName_LastName
ACCEPT userid PROMPT 'Enter the name of your Oracle userid: '
ACCEPT passwd PROMPT 'Enter the password for your Oracle userid: '
--  Create the userid
DROP USER &userid cascade;
CREATE user &userid IDENTIFIED BY &passwd 
DEFAULT TABLESPACE users
TEMPORARY TABLESPACE TEMP;
-- Grant the DBA Role
GRANT DBA TO &userid WITH ADMIN OPTION;
-- Grant the role CONNECT
GRANT CONNECT TO &userid WITH ADMIN OPTION;
-- Grant the role RESOURCE
GRANT RESOURCE TO &userid WITH ADMIN OPTION; 
-- Set the default roles
ALTER user &userid DEFAULT ROLE DBA, CONNECT, RESOURCE;
-- Grant the SYSTEM PRIVILEGE CREATE SESSION
GRANT CREATE SESSION TO &userid WITH ADMIN OPTION;
select * from dba_users where username = upper('&userid');
undefine userid;
undefine passwd;
