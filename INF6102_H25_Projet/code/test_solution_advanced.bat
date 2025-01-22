@echo off
set agent=advanced
@echo RUNNING Agent:%agent%
@echo #############################

set year=%DATE:~0,4%
set month=%DATE:~5,2%
set day=%DATE:~8,2%
set hour=%TIME:~0,2%
IF "%HOUR:~0,1%" == " " SET HOUR=0%HOUR:~1,1%
set minute=%TIME:~3,2%
set second=%TIME:~6,2%
set datetime=%year%_%month%_%day%_%hour%_%minute%_%second%
set foldername=%agent%_%datetime%

if not exist results mkdir results
if not exist results\%foldername% mkdir results\%foldername%

set instance=eternity_trivial_A
@echo Starting Instance:%instance%
python main.py --agent=%agent% --infile=instances/%instance%.txt --outfile=results/%foldername%/%instance%.txt --visufile=results/%foldername%/%instance%.png >> results\%foldername%\output.txt
@echo Ended Instance:%instance%
@echo ####################

set instance=eternity_trivial_B
@echo Starting Instance:%instance%
python main.py --agent=%agent% --infile=instances/%instance%.txt --outfile=results/%foldername%/%instance%.txt --visufile=results/%foldername%/%instance%.png >> results\%foldername%\output.txt
@echo Ended Instance:%instance%
@echo ####################

set instance=eternity_A
@echo Starting Instance:%instance%
python main.py --agent=%agent% --infile=instances/%instance%.txt --outfile=results/%foldername%/%instance%.txt --visufile=results/%foldername%/%instance%.png >> results\%foldername%\output.txt
@echo Ended Instance:%instance%
@echo ####################

set instance=eternity_B
@echo Starting Instance:%instance%
python main.py --agent=%agent% --infile=instances/%instance%.txt --outfile=results/%foldername%/%instance%.txt --visufile=results/%foldername%/%instance%.png >> results\%foldername%\output.txt
@echo Ended Instance:%instance%
@echo ####################

set instance=eternity_C
@echo Starting Instance:%instance%
python main.py --agent=%agent% --infile=instances/%instance%.txt --outfile=results/%foldername%/%instance%.txt --visufile=results/%foldername%/%instance%.png >> results\%foldername%\output.txt
@echo Ended Instance:%instance%
@echo ####################

set instance=eternity_D
@echo Starting Instance:%instance%
python main.py --agent=%agent% --infile=instances/%instance%.txt --outfile=results/%foldername%/%instance%.txt --visufile=results/%foldername%/%instance%.png >> results\%foldername%\output.txt
@echo Ended Instance:%instance%
@echo ####################

set instance=eternity_E
@echo Starting Instance:%instance%
python main.py --agent=%agent% --infile=instances/%instance%.txt --outfile=results/%foldername%/%instance%.txt --visufile=results/%foldername%/%instance%.png >> results\%foldername%\output.txt
@echo Ended Instance:%instance%
@echo ####################

set instance=eternity_complet
@echo Starting Instance:%instance%
python main.py --agent=%agent% --infile=instances/%instance%.txt --outfile=results/%foldername%/%instance%.txt --visufile=results/%foldername%/%instance%.png >> results\%foldername%\output.txt
@echo Ended Instance:%instance%
@echo ####################


type results\%foldername%\output.txt

@echo ##################################
@echo ### FINISHED                   ###
@echo ##################################
@pause