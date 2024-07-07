import os
import torch
import time
import argparse
from torch import nn
import numpy as np
import pandas as pd
from random import randrange
from random import randint
from nanopore_models import bnLSTM_32window


def Signalstart(Signal):
	Start_point, Pro_start, Pre_start = 0, [], []
	Signal_lst = Signal.tolist()
	Start_point = (Signal_lst.index(max(Signal_lst[10:3000])),max(Signal_lst[10:3000]))
	Pre_start = Signal_lst[Start_point[0]-19:Start_point[0]-1]
	Pro_start = Signal_lst[Start_point[0]+1:Start_point[0]+19]
	if not Pro_start or not Pre_start:
		return int("0")
	if Start_point[1] > sum(Pre_start)/len(Pre_start):
		if Start_point[1] > sum(Pro_start)/len(Pro_start):
			if np.var(Signal[Start_point[0]+50:Start_point[0]+80]) > np.var(Signal[Start_point[0]-80:Start_point[0]-50]):
				return(Start_point[0])
	## if couldnt find valid start poin then return 0 as start point
	return int("0")


def differences_transform(signal):
	return np.diff(signal)


def cut_patchs(signal, seq_length, stride, patch_size):
	split_signal = np.zeros((seq_length, patch_size), dtype="float32")
	for i in range(seq_length):
		split_signal[i, :] = signal[(i*stride):(i*stride)+patch_size]
	return split_signal


def myprint(string, log):
	log.write(string+'\n')
	print(string)


def inference(log, device, reads, model, batch_size, length, is_patchs, 
			  seq_length, stride, patch_size, label):
	model.eval()
	with torch.no_grad():
		read_count, start_zero, short_reads, accepted_reads = 0, 0, 0, 0
		true_pred, false_pred = 0, 0
		batch_count, total_time = 0, 0
		segment_arr = []

		for read in reads:
			read_count += 1
			start = Signalstart(read)
			if start == 0:
				start_zero += 1
				continue
			read = differences_transform(read[start:])
			if len(read) < length:
				short_reads += 1
				continue

			accepted_reads += 1
			read = read[: length]
			if is_patchs:
				segment = cut_patchs(read, seq_length, stride, patch_size)
			segment_arr.append(segment)

			if accepted_reads % batch_size == 0 and accepted_reads != 0:
				batch_count += 1
				segment_arr = np.array(segment_arr)
				inputs = torch.FloatTensor(segment_arr).to(device)
				start_time = time.time()
				outputs = model(inputs)
				end_time = time.time()
				print(end_time - start_time)
				if batch_count > 3:
					total_time += end_time - start_time

				outputs = outputs.max(dim=1).indices
				for y in outputs:
					if y == label:
						true_pred += 1
					else:
						false_pred += 1

				del segment_arr
				segment_arr = []
				myprint(f'total reads: {read_count}, accepted reads: {accepted_reads}, start zero: {start_zero}, short reads: {short_reads}, true predicts: {true_pred}, false predicts: {false_pred}', log)

	segment_arr = np.array(segment_arr)
	inputs = torch.FloatTensor(segment_arr).to(device)
	outputs = model(inputs)
	outputs = outputs.max(dim=1).indices
	for y in outputs:
		if y == label:
			true_pred += 1
		else:
			false_pred += 1
	myprint(f'total reads: {read_count}, accepted reads: {accepted_reads}, start zero: {start_zero}, short reads: {short_reads}, true predicts: {true_pred}, false predicts: {false_pred}', log)
	return true_pred, false_pred, start_zero + short_reads, total_time / (batch_count - 3)


if __name__ == '__main__':
	# Get command arguments
	parser = argparse.ArgumentParser(description="Test model")
	parser.add_argument("--pos_dataset", '-pos', type=str, required=True, help="Path to the positive dataset (a npy file)")
	parser.add_argument("--neg_dataset", '-neg', type=str, required=True, help="Path to the negative dataset (a npy file)")
	parser.add_argument("--model_state", '-ms', type=str, required=True, help="Path of the model")
	parser.add_argument("--batch_size", '-b', type=int, default=1, help="Batch size, default 1")
	parser.add_argument("--length", '-len', type=int, default=2100, help="The length of each signal segment, default 2100")
	parser.add_argument("--is_patchs", '-ip', type=bool, default=True, help="Convert electrical signals into patchs, default True")
	parser.add_argument("--seq_length", '-sl', type=int, default=200, help="Sequence length after patch, default 200")
	parser.add_argument("--stride", '-s', type=int, default=10, help="Patch step size, default 10")
	parser.add_argument("--patch_size", '-ps', type=int, default=32, help="The size of patch, default 32")
	parser.add_argument("--gpu_ids", '-g', type=str, default='0', help="Specify the GPU to use, if not specified, use all or cpu, default 0")
	args = parser.parse_args()

	# output file
	folder, _ = os.path.split(args.model_state)
	log = open(folder+'/inference.txt', mode='w', encoding='utf-8')	

	# Print parameter information
	for arg in vars(args):
		myprint(f"{arg}: {getattr(args, arg)}", log)
		
	model = torch.load(args.model_state)
	# Use GPU or CPU
	if args.gpu_ids:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
	device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
		myprint(f"Test in {device}", log)
	else:
		myprint(f"Test in CPU", log)
	model.to(device)

	# Load dataset and testing
	# reads = np.array(pd.read_csv(args.pos_dataset, header=None, dtype=np.int16))
	reads = np.load(args.pos_dataset, allow_pickle=True)
	tp, fn, pos_bad, pos_infer_time = inference(log, device, reads, model, args.batch_size, args.length, 
	   args.is_patchs, args.seq_length, args.stride, args.patch_size, 0)
	
	# reads = np.array(pd.read_csv(args.neg_dataset, header=None, dtype=np.int16))
	reads = np.load(args.neg_dataset, allow_pickle=True)
	tn, fp, neg_bad, neg_infer_time = inference(log, device, reads, model, args.batch_size, args.length, 
	   args.is_patchs, args.seq_length, args.stride, args.patch_size, 1)

	myprint(f'TP: {tp}, FN: {fn}, TN: {tn}, FP: {fp}, pos bad reads: {pos_bad}, neg bad reads: {neg_bad}, pos inference time: {pos_infer_time}, neg inference time: {neg_infer_time}', log)

	# Calculate evaluation index values
	aver_infer_time = (pos_infer_time + neg_infer_time) / 2
	accuracy = (tp + tn + neg_bad) / (tp + tn + fp + fn + pos_bad + neg_bad)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn + pos_bad)
	f1_score = (2 * precision * recall) / (precision + recall)
	myprint('with bad reads', log)
	myprint('accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1_score: {:.4f}, average inference time: {:.4f}'.format(
		accuracy, precision, recall, f1_score, aver_infer_time), log)
	log.close()
