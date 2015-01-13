hypoTestFileName='data/oHl_300_hypo_Dev.matrix';
visibleSize=300; 

thetaFileName='data/thetas/theta_v300_h50_r0.01_l0.0001_b3.matrix';
hiddenSize=50;
test(thetaFileName, hypoTestFileName, hiddenSize, visibleSize)

thetaFileName='data/thetas/theta_v300_h100_r0.01_l0.0001_b3.matrix';
hiddenSize=100;
test(thetaFileName, hypoTestFileName, hiddenSize, visibleSize)

thetaFileName='data/thetas/theta_v300_h500_r0.01_l0.0001_b3.matrix';
hiddenSize=500;
test(thetaFileName, hypoTestFileName, hiddenSize, visibleSize)

thetaFileName='data/thetas/theta_v300_h1000_r0.01_l0.0001_b3.matrix';
hiddenSize=1000;
test(thetaFileName, hypoTestFileName, hiddenSize, visibleSize)

thetaFileName='data/thetas/theta_v300_h5000_r0.01_l0.0001_b3.matrix';
hiddenSize=5000;
test(thetaFileName, hypoTestFileName, hiddenSize, visibleSize)
