#!/bin/sh

. /projects/academic/mdupuis2/pavan/bin/py2019_load.sh

cd step2_1i

for i in {5..196}
do
cd run_${i}_*
cp /projects/academic/mdupuis2/pavan/runs_aflow/INCAR* INCAR
cp /projects/academic/mdupuis2/pavan/runs_aflow/KPOINT* KPOINTS
cp /projects/academic/mdupuis2/pavan/runs_aflow/slurmscript* slurmscript
sed -i -e "s/job-name=/job-name=run_${i}/g" slurmscript
python /projects/academic/mdupuis2/pavan/bin/super_from_poscar.py
python /projects/academic/mdupuis2/pavan/bin/step2_jobbuilder.py
sbatch slurmscript
cd ..
echo "run_${i}"
done

