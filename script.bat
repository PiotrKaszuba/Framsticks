(FOR %%G IN (1, 2, 3) DO FOR /L %%N IN (1,1,10) DO (echo start & wmic os get localdatetime & echo. & frams "im average.sim" "World.wrldsiz = 20.0 + (%%G - 1) * 0.001;" "if(%%G == 3) { ExpProperties.p_xov = 0; ExpProperties.p_mut = 72; ExpProperties.selrule = 2; ExpProperties.p_nop = 28; ExpProperties.capacity = 200; }" "Math.randomize();" "Simulator.init();" "var tempVarI = 1; while (tempVarI<50) { GenePools[0].add(ExpProperties.initialgen); tempVarI += 1; }"  "Simulator.start();""while (ExpState.totaltestedcr<25000 && Simulator.running) Simulator.step();" "sa finished_%%G_%%N.expt" "-q" & echo finish & wmic os get localdatetime & echo. )) > outputFile.txt
