def MR(rank):
    return float(sum(rank)/len(rank))

def MRR(rank):
    reciprocal_rank = [float(1.0/x) for x in rank]
    return float(sum(reciprocal_rank)/len(reciprocal_rank))

def hits_k(rank, k):
    result = 0
    for i in rank:
        if i <= k:
            result += 1
    return float(result/len(rank))

def metrics(rank, k_vals=(1, 5, 10)):
    result = {}
    mr = MR(rank)
    result["mr"] = mr
    mrr = MRR(rank)
    result["mrr"] = mrr
    for k in k_vals:
        result["hits@{k}".format(k=k)] = hits_k(rank, k)
    return result