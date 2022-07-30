import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Qt5Agg")

# PLOT TEXT-BASED DEEP LEARNING MODEL ACCURACIES
MLPwithReLu = [55.50,	51.2]
MLPwithTanh	= [58.2, 48.01]
MLPwithSoftMax = [47.75,	48.01]
MLPwithSigmoid = [	52.25,	52.0]
LSTMwithReLu = [51.35, 52.0]
LSTMwithTanh = [52.25, 52.0]
LSTMwithSoftMax	= [51.35, 52.0]
LSTMwithSigmoid	= [53.15, 52.0]
GRUwithReLu	= [52.25, 52.0]
GRUwithTanh	= [50.45, 52.0]
GRUwithSoftMax = [55.86, 60.0]
GRUwithSigmoid = [52.25, 52.0]
SRNNwithReLu = [50.45,	52.0]
SRNNwithTanh = [49.55,	42.0]
SRNNwithSoftMax	= [46.85, 50.0]
SRNNwithSigmoid = [	52.25, 52.0]
DensewithReLu = [82.88,	68.0]
DensewithTanh = [53.15,	52.0]
DensewithSoftMax = [57.66, 52.0]
DensewithSigmoid = [52.25, 52.0]

textBasedTrainingMean = MLPwithReLu[0] + MLPwithTanh[0] + MLPwithSoftMax[0]  + MLPwithSigmoid[0] + LSTMwithReLu[0] +  LSTMwithTanh[0] + LSTMwithSoftMax[0] + LSTMwithSigmoid[0] + GRUwithReLu[0] + GRUwithTanh[0] + GRUwithSoftMax[0] + GRUwithSigmoid[0] + SRNNwithReLu[0] + SRNNwithTanh[0] + SRNNwithSoftMax[0] + SRNNwithSigmoid[0] + DensewithReLu[0] + DensewithTanh[0] + DensewithSoftMax[0] + DensewithSigmoid[0]
textBasedValidationMean = MLPwithReLu[1] + MLPwithTanh[1] + MLPwithSoftMax[1]  + MLPwithSigmoid[1] + LSTMwithReLu[1] +  LSTMwithTanh[1] + LSTMwithSoftMax[1] + LSTMwithSigmoid[1] + GRUwithReLu[1] + GRUwithTanh[1] + GRUwithSoftMax[1] + GRUwithSigmoid[1] + SRNNwithReLu[1] + SRNNwithTanh[1] + SRNNwithSoftMax[1] + SRNNwithSigmoid[1] + DensewithReLu[1] + DensewithTanh[1] + DensewithSoftMax[1] + DensewithSigmoid[1]
treeBasedTrainingMean = textBasedTrainingMean/20.0
treeBasedValidationMean = textBasedValidationMean/20.0

print("Avergae TA from Text-Based Deep Learning Models:", treeBasedTrainingMean)
print("Avergae VA from Text-Based Deep Learning Models:", treeBasedValidationMean)

xAxisLabels = ["Training Accuracy", "Validation Accuracy"]

fig, plot = plt.subplots(1)

plot.plot(MLPwithReLu, label="MLPwithReLu", marker="o")
plot.plot(MLPwithTanh, label="MLPwithTanh", marker="o")
plot.plot(MLPwithSoftMax, label="MLPwithSoftMax", marker="o")
plot.plot(MLPwithSigmoid, label="MLPwithSigmoid", marker="o")

plot.plot(LSTMwithReLu, label="LSTMwithReLu", marker="o")
plot.plot(LSTMwithTanh, label="LSTMwithTanh", marker="o")
plot.plot(LSTMwithSoftMax, label="LSTMwithSoftMax", marker="o")
plot.plot(LSTMwithSigmoid, label="LSTMwithSigmoid", marker="o")
plot.plot(GRUwithReLu, label="GRUwithReLu", marker="o")
plot.plot(GRUwithTanh, label="GRUwithTanh", marker="o")
plot.plot(GRUwithSoftMax, label="GRUwithSoftMax", marker="o")
plot.plot(GRUwithSigmoid, label="GRUwithSigmoid", marker="o")
plot.plot(SRNNwithReLu, label="SRNNwithReLu", marker="o")
plot.plot(SRNNwithTanh, label="SRNNwithTanh", marker="o")
plot.plot(SRNNwithSoftMax, label="SRNNwithSoftMax", marker="o")
plot.plot(SRNNwithSigmoid, label="SRNNwithSigmoid", marker="o")
plot.plot(DensewithReLu, label="DensewithReLu", marker="o")
plot.plot(DensewithTanh, label="DensewithTanh", marker="o")
plot.plot(DensewithSoftMax, label="DensewithSoftMax", marker="o")
plot.plot(DensewithSigmoid, label="DensewithSigmoid", marker="o")

plot.legend(loc="upper left")
xTickValues = [0, 1]
plt.title("Text-Based Deep Learning Model Results")
plt.xlabel('Accuracy Type')
plt.ylabel('Accuracy Score')
plt.xticks(ticks=xTickValues, labels=xAxisLabels)
plt.show()


# PLOT TREE-BASED DEEP LEARNING MODEL ACCURACIES USING UNHASHED NODES
fig2, plot2 = plt.subplots(1)

MLPwithReLu = [52.66, 57.58]
MLPwithTanh	= [49.30, 48.48]
MLPwithSoftMax = [49.22, 42.42]
MLPwithSigmoid = [50.88, 57.58]

LSTMwithReLu = [56.25, 75.76]
LSTMwithTanh = [60.94, 70.00]
LSTMwithSoftMax	= [72.66, 63.64]
LSTMwithSigmoid	= [77.34, 70.00]

GRUwithReLu	= [71.88, 66.67]
GRUwithTanh	= [53.12, 51.52]
GRUwithSoftMax = [74.22, 60.61]
GRUwithSigmoid = [76.56, 75.76]

SRNNwithReLu = [61.72, 48.48]
SRNNwithTanh = [71.09,	66.67]
SRNNwithSoftMax	= [100.0, 60.61]
SRNNwithSigmoid = [	100.0, 70.0]

DensewithReLu = [50.78,	57.58]
DensewithTanh = [50.78,	57.58]
DensewithSoftMax = [96.88, 75.76]
DensewithSigmoid = [97.66, 57.58]


plot2.plot(MLPwithReLu, label="MLPwithReLu", marker="o")
plot2.plot(MLPwithTanh, label="MLPwithTanh", marker="o")
plot2.plot(MLPwithSoftMax, label="MLPwithSoftMax", marker="o")
plot2.plot(MLPwithSigmoid, label="MLPwithSigmoid", marker="o")

plot2.plot(LSTMwithReLu, label="LSTMwithReLu", marker="o")
plot2.plot(LSTMwithTanh, label="LSTMwithTanh", marker="o")
plot2.plot(LSTMwithSoftMax, label="LSTMwithSoftMax", marker="o")
plot2.plot(LSTMwithSigmoid, label="LSTMwithSigmoid", marker="o")
plot2.plot(GRUwithReLu, label="GRUwithReLu", marker="o")
plot2.plot(GRUwithTanh, label="GRUwithTanh", marker="o")
plot2.plot(GRUwithSoftMax, label="GRUwithSoftMax", marker="o")
plot2.plot(GRUwithSigmoid, label="GRUwithSigmoid", marker="o")
plot2.plot(SRNNwithReLu, label="SRNNwithReLu", marker="o")
plot2.plot(SRNNwithTanh, label="SRNNwithTanh", marker="o")
plot2.plot(SRNNwithSoftMax, label="SRNNwithSoftMax", marker="o")
plot2.plot(SRNNwithSigmoid, label="SRNNwithSigmoid", marker="o")
plot2.plot(DensewithReLu, label="DensewithReLu", marker="o")
plot2.plot(DensewithTanh, label="DensewithTanh", marker="o")
plot2.plot(DensewithSoftMax, label="DensewithSoftMax", marker="o")
plot2.plot(DensewithSigmoid, label="DensewithSigmoid", marker="o")
plot2.legend(loc="upper left")
xTickValues = [0, 1]
plt.title("Tree-Based Deep Learning Model with Segmentation Results")
plt.xlabel('Accuracy Type')
plt.ylabel('Accuracy Score')
plt.xticks(ticks=xTickValues, labels=xAxisLabels)
plt.show()

