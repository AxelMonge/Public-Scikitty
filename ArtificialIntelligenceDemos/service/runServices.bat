@echo off

REM Lanzar SWI-Prolog en segundo plano
start "" swipl -s service_server.pl

REM Esperar unos segundos para asegurarse de que SWI-Prolog se inicie completamente
timeout /t 3 /nobreak >nul

REM Ejecutar Django en primer plano
python manage.py runserver