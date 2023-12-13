import torch
import numpy as np
import random
import time
import os
import argparse
import string
import networkx as nx
from tqdm import tqdm
import wandb
import json
import multiprocessing
from collections import defaultdict

from data import load_data
from batch import batch, negative_sampling
from feature_init import heuristics_init
from model import FastISDEA
from metrics import metrics
from utils import create_hash, wandb_run_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast_ISDEA.")
    parser.add_argument(
        "--exp_name", type=str, required=True, help="Name of the experiment. Used for logging purposes."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset task (excluding the -trans or -ind suffix)"
    )
    parser.add_argument(
        "--dataset_folder", type=str, default="data/", help="Dataset directory"
    )
    parser.add_argument(
        "--negative_sampling", type=str, default="node", choices=["relation", "node", "relation_full", "node_full"], 
        help="Negatively sample relation or nodes, or take entire graph as negative samples for evaluation"
    )
    parser.add_argument(
        "--seed", type=int, default=46, help="random seed"
    )
    parser.add_argument(
        "--mode", type=str, default="traintest", choices=["train", "test", "traintest"], help="traintest or train or test"
    )
    parser.add_argument(
        "--num_cpus", type=int, default=multiprocessing.cpu_count(), 
        help="Number of CPUs to compute shortest path distances"
    )
    parser.add_argument(
        "--load_ckpt", type=str, required=False, help="load pretrained model"
    )
    parser.add_argument(
        "--main_folder", type=str, default="./", help="path of source code"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001
    )
    parser.add_argument(
        "--epoch", type=int, default=51
    )
    parser.add_argument(
        "--valid_epoch", type=int, default=2, help="validation per valid_epoch epochs"
    )
    parser.add_argument(
        "--sign_k", type=int, default=3
    )
    parser.add_argument(
        "--init_feature_dim", type=int, default=16
    )
    parser.add_argument(
        "--predictor_feature_hidden", type=int, default=128
    )
    parser.add_argument(
        "--batchsize", type=int, default=256
    )
    parser.add_argument(
        "--negative_tail_train", type=int, default=2
    )
    parser.add_argument(
        "--negative_relation_train", type=int, default=2
    )
    parser.add_argument(
        "--hits_k", type=str, default="1,5,10", help="Comma-separated list of k values for hits@k metric"
    )
    parser.add_argument(
        "--early_stop_hits_k", type=int, default=10, 
        help="Which hits@k metric to use for early stopping. Must be one of the values in --hits_k."
    )
    # Support Weight & Biases logging
    parser.add_argument(
        "--wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, required=False, help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, required=False, help="Weights & Biases team name/account username"
    )
    parser.add_argument(
        "--wandb-job-type", type=str, required=False, help="Weights & Biases job type"
    )

    args = parser.parse_args()

    do_train = args.mode == "traintest" or args.mode == "train"
    do_test = args.mode == "traintest" or args.mode == "test"

    print(f"do_train: {do_train}, do_test: {do_test}")

    # Initialize Weights & Biases logging, and log the arguments for this run
    run_config = vars(args)
    run_hash = create_hash(str(vars(args)))
    run_config["run_hash"] = run_hash
    run_config["stage"] = args.mode
    wandb.init(mode="online" if args.wandb else "disabled",  # Turn on wandb logging only if --wandb is set
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type=args.wandb_job_type,
            config=run_config)
    # Custom run name: run hash + model name + task/dataset name
    wandb.run.name = wandb_run_name(run_hash, "train")
    print(f"\n### Run Hash: {run_hash} ###\n")

    # Set global random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_name = args.exp_name
    main_folder = args.main_folder

    data_folder = args.dataset_folder
    dataset_name = args.dataset
    
    result_folder = "result"
    output_folder = os.path.join(main_folder, result_folder, exp_name, dataset_name, run_hash)
    print(f"Results and model checkpoints will be saved to {output_folder}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, "result.txt")
    output_config = os.path.join(output_folder, "config.json")
    # Save the arguments for this run to a JSON file
    with open(output_config, "w") as f:
        json.dump(run_config, f, indent=4)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading data: data directory = {data_folder}, dataset = {dataset_name}")

    # train_train_data, train_valid_data, inf_observe_data, inf_test_data = load_data(dataset_name, data_folder)

    if do_train:
        train_train_data, train_valid_data = load_data(dataset_name, data_folder, mode="train")

        train_msg_edges = train_train_data["edge_list_mes"]
        train_sup_edges = train_train_data["edge_list_sup"]
        valid_edge = train_valid_data["edge_list"]
        G_valid = train_valid_data["G"]

        num_node_train = train_train_data["num_node"]
        num_relation_train = train_train_data["num_relation"]
        G_train_mes = train_train_data["G_mes"]
        G_train_mes_di = train_train_data["G_mes_di"]
        G_train_sup = train_train_data["G_sup"]
        train_sup_edge =  train_train_data["edge_list_sup"]
        adj_train = train_train_data["adj"]
        adjs_train = train_train_data["adjs"]

    if do_test:
        inf_observe_data, inf_test_data = load_data(dataset_name, data_folder, mode="test")

        inf_msg_edges = inf_observe_data["edge_list"]
        inf_test_edge = inf_test_data["edge_list"]
        G_test = inf_test_data["G"]

        num_node_inf = inf_observe_data["num_node"]
        num_relation_inf = inf_observe_data["num_relation"]
        G_inf = inf_observe_data["G"]
        G_inf_di = inf_observe_data["G_di"]
        adj_inf = inf_observe_data["adj"]
        adjs_inf = inf_observe_data["adjs"]
    
    print(f"Computing shortest path distance for {dataset_name} with {args.num_cpus} CPUs")

    if args.num_cpus <= 1:
        # Expose distance computation for each node to tqdm to show progress bar
        if do_train:
            dis_train = {
                node: nx.single_source_dijkstra_path_length(G_train_mes_di, node)
                for node in tqdm(G_train_mes_di.nodes, desc="Training graph")
            }
        if do_test:
            dis_inf = {
                node: nx.single_source_dijkstra_path_length(G_inf_di, node)
                for node in tqdm(G_inf_di.nodes, desc="Inference graph")
            }
    else:
        # Use multiprocessing to compute distances in parallel. 
        def compute_shortest_paths(args):
            G, node = args
            return node, nx.single_source_dijkstra_path_length(G, node)
        
        if do_train:
            dis_train = {}
            with multiprocessing.Pool(args.num_cpus) as pool:
                tasks = [(G_train_mes_di, node) for node in G_train_mes_di.nodes]
                results = pool.imap_unordered(compute_shortest_paths, tasks, chunksize=2000)
                with tqdm(total=len(G_train_mes_di.nodes), desc="Training graph") as pbar:
                    for node, result in results:
                        dis_train[node] = result
                        pbar.update()
        
        if do_test:
            dis_inf = {}
            with multiprocessing.Pool(args.num_cpus) as pool:
                tasks = [(G_inf_di, node) for node in G_inf_di.nodes]
                results = pool.imap_unordered(compute_shortest_paths, tasks, chunksize=2000)
                with tqdm(total=len(G_inf_di.nodes), desc="Inference graph") as pbar:
                    for node, result in results:
                        dis_inf[node] = result
                        pbar.update()
    
    batchsize = args.batchsize
    negative_tail_train = args.negative_tail_train
    negative_relation_train = args.negative_relation_train
    if args.negative_sampling == "relation":
        # result_folder = "result/fast_relation/"
        negative_relation_eval = 50
        negative_tail_eval = 0
    elif args.negative_sampling == "node":
        # result_folder = "result/fast_node/"
        negative_relation_eval = 0
        negative_tail_eval = 50
    elif args.negative_sampling == "relation_full":
        negative_relation_eval = num_relation_inf
        negative_tail_eval = 0
    elif args.negative_sampling == "node_full":
        negative_relation_eval = 0
        negative_tail_eval = num_node_inf

    print("Start negative sampling and preprocessing")

    # Compute head_rel2tail map to avoid sampling false negative tails
    # Compute head_tail2rel map to avoid sampling false negative relations
    if do_train:
        train_edges_total = train_msg_edges + train_sup_edges
        train_head_rel2tail = defaultdict(set)
        train_head_tail2rel = defaultdict(set)
        for edge in train_edges_total:
            head = edge[0]
            tail = edge[1]
            rel = edge[2]
            train_head_rel2tail[(head, rel)].add(tail)
            train_head_tail2rel[(head, tail)].add(rel)
        valid_edges_total = train_msg_edges + train_sup_edges + valid_edge
        valid_head_rel2tail = defaultdict(set)
        valid_head_tail2rel = defaultdict(set)
        for edge in valid_edges_total:
            head = edge[0]
            tail = edge[1]
            rel = edge[2]
            valid_head_rel2tail[(head, rel)].add(tail)
            valid_head_tail2rel[(head, tail)].add(rel)
    if do_test:
        test_edges_total = inf_msg_edges + inf_test_edge
        test_head_rel2tail = defaultdict(set)
        test_head_tail2rel = defaultdict(set)
        for edge in test_edges_total:
            head = edge[0]
            tail = edge[1]
            rel = edge[2]
            test_head_rel2tail[(head, rel)].add(tail)
            test_head_tail2rel[(head, tail)].add(rel)

    # Negative sampling
    if do_train:
        mixed_train_edge = negative_sampling(
            train_sup_edge, 
            train_head_rel2tail, 
            train_head_tail2rel, 
            num_node_train, 
            num_relation_train, 
            negative_tail_train, 
            negative_relation_train, 
            "train",
            full_graph=False
        )
        train_batches = batch(mixed_train_edge, num_relation_train, batchsize)
        mixed_train_data = []
        for train_batch in tqdm(train_batches, desc="Sampling in training batches"):
            chosen_relations = [train_batch[0][2].item()]
            heuristics = heuristics_init(G_train_mes, G_train_mes_di, dis_train, train_batch).to(device)
            mixed_train_data.append({"batch":train_batch, "chosen_relations":chosen_relations, "heuristics":heuristics})

        valid_batches = batch(valid_edge, num_relation_train, batchsize)
        mixed_valid_data = []
        for valid_batch in tqdm(valid_batches, desc="Sampling in validation batches"):
            valid_batch = negative_sampling(
                valid_batch, 
                valid_head_rel2tail,
                valid_head_tail2rel, 
                num_node_train, 
                int(num_relation_train/2), 
                negative_tail_eval, 
                negative_relation_eval, 
                "test",
                full_graph=args.negative_sampling.endswith("_full")
            )
            if valid_batch != []:
                heuristics = heuristics_init(G_train_mes, G_train_mes_di, dis_train, valid_batch).to(device)
                mixed_valid_data.append({"batch":valid_batch, "heuristics":heuristics})

    if do_test:
        test_batches = batch(inf_test_edge, num_relation_inf, batchsize)
        mixed_test_data = []
        for test_batch in tqdm(test_batches, desc="Sampling in test batches"):
            test_batch = negative_sampling(
                test_batch, 
                test_head_rel2tail, 
                test_head_tail2rel, 
                num_node_inf, 
                int(num_relation_inf/2), 
                negative_tail_eval, 
                negative_relation_eval, 
                "test",
                full_graph=args.negative_sampling.endswith("_full")
            )
            if test_batch != []:
                heuristics = heuristics_init(G_inf, G_inf_di, dis_inf, test_batch).to(device)
                mixed_test_data.append({"batch":test_batch, "heuristics":heuristics})

    lr = args.lr
    epoch = args.epoch
    sign_k = args.sign_k
    init_feature_dim = args.init_feature_dim
    predictor_feature_hidden = args.predictor_feature_hidden
    input_heuristics_dim = 2

    if do_train:
        train_input_node_feature = torch.ones(num_node_train, init_feature_dim).to(device)
        train_input_relation_feature = torch.ones(num_relation_train, init_feature_dim).to(device)

    if do_test:
        inf_input_node_feature = torch.ones(num_node_inf, init_feature_dim).to(device)
        inf_input_relation_feature = torch.ones(num_relation_inf, init_feature_dim).to(device)

    model = FastISDEA(input_heuristics_dim, init_feature_dim, predictor_feature_hidden, sign_k, device).to(device)

    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    valid_epoch = args.valid_epoch
    best_hits10 = 0
    best_mr = 1 + negative_tail_eval + negative_relation_eval
    best_epoch = 0
    early_stop = 0
    best_model_path = output_folder + "/model.pt" 

    hits_k = tuple(int(k) for k in args.hits_k.split(","))
    early_stop_hits_k = f"hits@{args.early_stop_hits_k}"

    if do_train:
        print("### Start training ###")
        for i in range(epoch + 1):
            # Skip first epoch, which is used for validation on the initial model
            if i > 0:
                model.train()
                random.shuffle(mixed_train_data)
                epoch_start_time = time.time()
                step_train_times = []
                for b in tqdm(mixed_train_data, desc=f"Training at Epoch {i}"):
                    train_batch, chosen_relations, heuristics = b["batch"], b["chosen_relations"], b["heuristics"]
                    step_start_time = time.time()
                    predicted, label = model.train_forward(train_input_relation_feature, train_input_node_feature, adj_train, adjs_train, heuristics, train_batch, chosen_relations, device)
                    loss = loss_fn(predicted, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step_end_time = time.time()
                    step_train_times.append(step_end_time - step_start_time)
                step_train_avg_time = sum(step_train_times) / len(step_train_times)
                epoch_end_time = time.time()
                epoch_time = epoch_end_time - epoch_start_time
                time_record = "Train Time " + str(i) + " " + str(epoch_time) + '\n'
                time_record = f"Train Time {i} {epoch_time}, avg step time: {step_train_avg_time}\n"
                with open(output_file, 'a') as f:
                    f.write(time_record)

            # Validation per valid_epoch epochs
            if i % valid_epoch == 0:
                with torch.no_grad():
                    valid_start_time = time.time()
                    model.eval()
                    chosen_relations = [a for a in range(num_relation_train)]
                    node_feature = model.process_node_feature(train_input_relation_feature, train_input_node_feature, adj_train, adjs_train, chosen_relations, device)
                    rank = []
                    for b in tqdm(mixed_valid_data, desc=f"Validation at Epoch {i}"):
                        valid_batch, heuristics = b["batch"], b["heuristics"]
                        batch_rank = model.test_forward(valid_batch, node_feature, heuristics, negative_tail_eval, negative_relation_eval, device)
                        rank += batch_rank
                    valid_end_time = time.time()

                result = metrics(rank, k_vals=hits_k)
                del node_feature
                torch.cuda.empty_cache()

                message = "Valid Results " + str(i) + " " + str(result) + '\n'
                message += "Valid Time " + str(i) + " " + str(valid_end_time - valid_start_time) + '\n'
                with open(output_file, 'a') as f:
                    f.write(message)
                if result["mr"] < best_mr:
                    print(f"--- Saving best model at epoch {i} ---")
                    early_stop = 0
                    best_mr = result["mr"]
                    best_epoch = i
                    torch.save(model, best_model_path)
                if result[early_stop_hits_k] > best_hits10:
                    best_hits10 = result[early_stop_hits_k]
                if result[early_stop_hits_k] < best_hits10 - 0.1:
                    early_stop += 1
                    if early_stop == 5:
                        print(f"--- Early stop at epoch {i} ---")
                        break

                # Print validation results to console
                print(f"Validation results at epoch {i}:")
                for k, v in result.items():
                    print(f"{k}: {v}")
                print("\n")
                # Log validation results to Weights & Biases, with prefix "valid/"
                wandb_valid_results = {f"valid/{k}": v for k, v in result.items()}
                wandb.log(wandb_valid_results, step=i)

        if do_test:
            print("### Start testing ###")
            model = torch.load(best_model_path)
            model.eval()
            chosen_relations = [a for a in range(num_relation_inf)]
            node_feature = model.process_node_feature(inf_input_relation_feature, inf_input_node_feature, adj_inf, adjs_inf, chosen_relations, device)
            rank = []

            with torch.no_grad():
                for b in mixed_test_data:
                    test_batch, heuristics = b["batch"], b["heuristics"]
                    batch_rank = model.test_forward(test_batch, node_feature, heuristics, negative_tail_eval, negative_relation_eval, device)
                    rank += batch_rank

                result = metrics(rank, k_vals=hits_k)
                del node_feature
                torch.cuda.empty_cache()

            message = "Test " + str(best_epoch) + " " + str(result) + '\n'
            with open(output_file, 'a') as f:
                f.write(message)

            # Log test results to Weights & Biases, with prefix "test/"
            wandb_test_results = {f"test/{k}": v for k, v in result.items()}
            wandb.log(wandb_test_results)

    else:
        print("### Start testing with loaded checkpoint ###")
        best_model_path = os.path.join(args.load_ckpt, "model.pt")
        test_start_time = time.time()
        model = torch.load(best_model_path)
        model.eval()
        chosen_relations = [a for a in range(num_relation_inf)]
        node_feature = model.process_node_feature(inf_input_relation_feature, inf_input_node_feature, adj_inf, adjs_inf, chosen_relations, device)
        rank = []

        with torch.no_grad():
            for b in tqdm(mixed_test_data, desc=f"Testing batch"):
                test_batch, heuristics = b["batch"], b["heuristics"]
                batch_rank = model.test_forward(test_batch, node_feature, heuristics, negative_tail_eval, negative_relation_eval, device)
                rank += batch_rank

            result = metrics(rank, k_vals=hits_k)
            del node_feature
            torch.cuda.empty_cache()

        test_end_time = time.time()
        test_time = test_end_time - test_start_time
        print(f"Test time: {test_time} seconds")

        message = "Test " + str(best_epoch) + " " + str(result) + '\n'
        with open(output_file, 'a') as f:
            f.write(message)

        # Print test results to console
        print(f"Test results:")
        for k, v in result.items():
            print(f"{k}: {v}")
        print("\n")

        # Log test results to Weights & Biases, with prefix "test/"
        wandb_test_results = {f"test/{k}": v for k, v in result.items()}
        wandb.log(wandb_test_results)
