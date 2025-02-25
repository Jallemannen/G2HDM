
import external.higgstools.python.Higgs.predictions as HP
import external.higgstools.python.Higgs.bounds as HB
import external.higgstools.python.Higgs.signals as HS

"""import sys
sys.path.append('/home/hjalmarh/Documents/2HDM_project/G2HDM_v1.1.0/external/higgstools/python')

import Higgs.Predictions as HP
import Higgs.Bounds as HB
import Higgs.Signals as HS"""

pred = HP.Predictions() # create the model predictions
bounds = HB.Bounds("/path/to/HBDataset") # load HB dataset
signals = HS.Signals("/path/to/HSDataset") # load HS dataset

# add a SM-like particle
h = pred.addParticle(HP.NeutralScalar("h", "even"))
h.setMass(125.09)
HP.effectiveCouplingInput(h, HP.smLikeEffCouplings)
# evaluate HiggsSignals
chisqSM = signals(pred)

# now give it some lepton-flavor violating decay
# there are very strong limits on this kind of process in HiggsBounds
h.setDecayWidth("emu", 1e-6)

# evaluate HiggsBounds
hbresult = bounds(pred)
print(hbresult)
# evaluate HiggsSignals
chisq = signals(pred)
print(f"HiggsSignals chisq: {chisq} compared to a SM chisq of {chisqSM}")