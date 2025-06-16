import pickle
import matplotlib.pyplot as plt

# with open('/workspace/Proof-of-concept/outputs_best_model/acuraccies.pkl','rb') as f:  # Python 3: open(..., 'rb')
#     acc_best = pickle.load(f)
    
# with open('/workspace/Proof-of-concept/outputs_weights_avg/acuraccies.pkl','rb') as f:  # Python 3: open(..., 'rb')
#     acc_avg = pickle.load(f)
    
# plt.rcParams["figure.figsize"] = (7,5)
# plt.plot(range(len(acc_best["valid_acc"])), acc_best["valid_acc"], label='Valid Acc Best Model', linestyle='--')
# plt.plot(range(len(acc_best["test_acc"])), acc_best["test_acc"], label='Test Acc Best Model', linestyle='--')
# plt.plot(range(len(acc_avg["valid_acc"])), acc_avg["valid_acc"], label='Valid Acc weights Avg', linestyle='--')
# plt.plot(range(len(acc_avg["test_acc"])), acc_avg["test_acc"], label='Test Acc Weights Avg', linestyle='--')

# plt.xlabel('Rounds')
# plt.ylabel('Accuracies')
# plt.ylim(0,1)
# plt.xlim(-0.5,len(acc_best["test_acc"])-1+0.5)
# plt.xticks(range(len(acc_best["valid_acc"])), range(1,len(acc_best["valid_acc"])+1))
# plt.title('Accuracies Comparison')
# plt.legend(loc='lower left')
# plt.savefig('/workspace/Proof-of-concept/foo.png')

with open('/workspace/Proof-of-concept/arch_1.pkl','rb') as f:  # Python 3: open(..., 'rb')
    acc_best = pickle.load(f)
    
print(acc_best)