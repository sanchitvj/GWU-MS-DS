SELECT last_name, salary
FROM   employees
WHERE  manager_id IN (SELECT employee_id
                     FROM   employees
                     WHERE  last_name = 'King');
