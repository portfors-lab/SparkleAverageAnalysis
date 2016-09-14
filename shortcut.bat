@echo off
setlocal EnableExtensions EnableDelayedExpansion
cls

:: --------------------------------------------------------
::
:: HOW TO USE:
:: To use this .bat file for running and updating Sparkle-
:: AverageAnalysi, make a copy of this file and rename it
:: something easy to remember (e.g. SparkleAverageAnalysis.bat).
:: Now in the copy of this file you will want to change the
:: variable of "location" (line 24) with the path to the
:: SparkleAverageAnalysis directory (where you found this file).
:: After you complete that you will want to change the
:: variable for "gitLocation" (line 28) with the path to
:: where your git.exe is stored. Once you have those set,
:: you can either create a shortcut to your newly edited
:: file or just run the .bat file. While running it with
:: this code, it will check for updates before running
:: and will ask the user if they wish to update if there
:: are any new updates.
::
:: Place your path to SparkleAverageAnalysis here
set location="C:\Users\Name\Documents\SparkleAverageAnalysis\"
cd %location%
::
:: Place your path to Git here
set gitLocation="C:\Program Files\Git\bin"
SET PATH=%PATH%;%gitLocation%
::
:: --------------------------------------------------------

title SparkleAverageAnalysis

:: Get versions of SparkleAverageAnalysis
git remote update
for /f "delims=" %%i in ('git rev-parse @{0}') do set local=%%i
for /f "delims=" %%i in ('git rev-parse origin/master') do set remote=%%i
for /f "delims=" %%i in ('git merge-base @ origin/master') do set base=%%i

echo.

:: Check relation of various versions
if "%local%" equ "%remote%" (
    echo Sparkle Average Analysis is up to date.
    goto :runSparkleAverageAnalysis
) else if "%local%" equ "%base%" (
    echo A newer version of Sparkle Average Analysis is avaliable.
) else if "%remote%" equ "%base%" (
    echo Your local branch of Sparkle Average Analysis is ahead of origin/master.
    goto :runSparkleAverageAnalysis
)

set "answer=%globalparam1%"
goto :answerCheck

:updatePrompt
set /p "answer=Update SparkleAverageAnalysis? (y or n): "
goto :answerCheck

:answerCheck
if not defined answer goto :updatePrompt

echo.

if "%answer%" == "y" (
    git pull
)

:runSparkleAverageAnalysis
echo.
python run.py