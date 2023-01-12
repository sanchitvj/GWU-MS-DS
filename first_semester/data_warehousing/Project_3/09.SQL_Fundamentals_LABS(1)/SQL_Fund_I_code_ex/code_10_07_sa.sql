INSERT INTO employees (employee_id, 
                 first_name, last_name, 
                 email, phone_number,
                 hire_date, job_id, salary, 
                 commission_pct, manager_id,
                 department_id)
VALUES		   (207, 
                 'Eva', 'Jones', 
                 'EJONES', '515.124.4666', 
                 SYSDATE, 'AC_ACCOUNT', 3300, 
                 NULL, 205, 70);


INSERT INTO departments(department_id, 
       department_name, manager_id, location_id)
VALUES (70, 'Public Relations', 100, 1700);

