SELECT employee_id,salary,last_name FROM employees M
WHERE EXISTS
(SELECT employee_id FROM employees W
 WHERE (W.manager_id=M.employee_id) AND w.salary > 10000);
