hypoTestFileName='data/oHl_hypo_Train.matrix';
visibleSize=50; 

thetaFileName='data/thetas/theta_v50_h50_r0.01_l0.0001_b3.matrix';
hiddenSize=50;
test_training_set(thetaFileName, hypoTestFileName, hiddenSize, visibleSize)

thetaFileName='data/thetas/theta_v50_h100_r0.01_l0.0001_b3.matrix';
hiddenSize=100;
test_training_set(thetaFileName, hypoTestFileName, hiddenSize, visibleSize)

thetaFileName='data/thetas/theta_v50_h500_r0.01_l0.0001_b3.matrix';
hiddenSize=500;
test_training_set(thetaFileName, hypoTestFileName, hiddenSize, visibleSize)

thetaFileName='data/thetas/theta_v50_h1000_r0.01_l0.0001_b3.matrix';
hiddenSize=1000;
test_training_set(thetaFileName, hypoTestFileName, hiddenSize, visibleSize)

thetaFileName='data/thetas/theta_v50_h5000_r0.01_l0.0001_b3.matrix';
hiddenSize=5000;
test_training_set(thetaFileName, hypoTestFileName, hiddenSize, visibleSize)
