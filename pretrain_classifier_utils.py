import numpy as np
import torch
import pandas as pd
from copy import deepcopy
import time
import csv
from utils import eval_metrics, auc_score


def train(model,n_cls, n_epochs, trainloader,valloader, criterion, optimizer, scheduler,tri ,device):
    
    best_model = deepcopy(model)
    best_f1 = -np.inf
    best_epoch = 0 
    for epoch in range(n_epochs):
        model.train()
        start_perf_counter = time.perf_counter()
        start_process_time = time.process_time()
        print(tri)
        #print(f'n_epoch:{epoch}, lr:{scheduler.get_last_lr()}')
        print(f'n_epoch:{epoch}')

        total_proba = []
        total_label = []
        running_loss = 0.0

        train_loss = 0.0
    
        for i,data in enumerate(trainloader, 0):

            inputs, labels = data['image'], data['label']
            inputs = inputs.to(device)

            if n_cls==1:
                labels = labels.unsqueeze(1).long().to(device)
            else:
                labels = labels.long().to(device)
            
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            
            
            if n_cls==1:
                proba = torch.sigmoid(outputs)
            else:
                proba = torch.nn.Softmax(dim=1)(outputs)
            
            total_proba.append(proba.detach().cpu().numpy())
            total_label.append(labels.detach().cpu().numpy())

            running_loss += loss.item()
            
            train_loss += loss.item()

            if i%100 == 99:
                print(f'[epoch_{epoch+1}, batch_{i+1}] loss: {running_loss/100}')
                running_loss = 0.0

        end_perf_counter = time.perf_counter()-start_perf_counter
        end_process_time = time.process_time()-start_process_time
        
    
        print(f'perf_counter : {end_perf_counter}')
        print(f'process_time : {end_process_time}')
        
        total_proba = np.concatenate(total_proba, 0)
        total_label = np.concatenate(total_label, 0)
       
        train_auc = auc_score(total_proba[:, 1], total_label)
        total_proba = np.argmax(total_proba,1)
        total_train_f1 = eval_metrics(total_proba, total_label)
        
        print('train_loss:',train_loss/len(trainloader), 'total_train_f1:',total_train_f1, 'train_auc', train_auc)

        valid_loss, valid_f1, probas, labels = test(model, n_cls,valloader, criterion, device)
        valid_auc = auc_score(probas[:, 1], labels)
        probas = np.argmax(probas,1)
        total_valid_f1 = eval_metrics(probas, labels)
        print('valid_loss:',valid_loss, 'valid_f1:',valid_f1, 'total_valid_f1', total_valid_f1, 'valid_auc', valid_auc )
        
        #torch.save(model.state_dict(), './model/'+tri+'_epoch'+str(epoch)+'.pth')
        scheduler.step(valid_loss)
        print('lr : ',optimizer.param_groups[0]['lr'])
        if epoch == 25:
            best_model = deepcopy(model)
            torch.save(model.state_dict(), '../weights/pretrain_model.pth')
            pass
    
    #torch.save(model.state_dict(), 'infer.pth')    

    return best_model, best_epoch


def test(model,n_cls, data_loader, criterion, device):
    model.eval()
    total_loss=0
    total_f1=0
    total_proba = []
    total_label = []
    with torch.no_grad():
        for i,data in enumerate(data_loader, 0):
            inputs, labels = data['image'], data['label']
            inputs = inputs.to(device)
            
            if n_cls==1:
                labels = labels.unsqueeze(1).long().to(device)
            else:
                labels = labels.long().to(device)

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            

            if n_cls==1:
                proba = torch.sigmoid(outputs)
            else:
                proba = torch.nn.Softmax(dim=1)(outputs)

            total_proba.append(proba.cpu().numpy())
            total_label.append(labels.cpu().numpy())
            proba = np.argmax(np.array(proba.cpu()),1)
            total_f1 += eval_metrics(proba,np.array( labels.cpu()))
            '''for idx,_ in enumerate(labels):
                print('predicted_labels:', torch.max(outputs.data, dim=1).indices[idx], 'label:', labels[idx].cpu())'''

        total_proba = np.concatenate(total_proba, 0)
        total_label = np.concatenate(total_label, 0)
    return total_loss/len(data_loader), total_f1/len(data_loader), total_proba, total_label
    
def submit(model,n_cls, file_name, data_loader, device):
    model.eval()
    
    results_df = pd.DataFrame()
    with torch.no_grad():
        for i,data in enumerate(data_loader, 0):
            inputs, image_name = data['image'], data['image_name']

            inputs = inputs.to(device)

            outputs = model.forward(inputs)
            

            if n_cls==1:
                proba = torch.sigmoid(outputs)
                #proba = np.where(proba.cpu() >= 0.5, 1, 0)
            else:
                proba = torch.nn.Softmax(dim=1)(outputs)
                #proba = np.argmax(np.array(proba.cpu()),1)


            #proba = torch.sigmoid(outputs)
            #proba = torch.nn.Softmax(dim=1)(outputs)
            #proba = np.where(proba.cpu() >= 0.5, 1, 0)
            #proba = np.argmax(np.array(proba.cpu()),1)
            
            
            for idx,_ in enumerate(inputs):
                
                if n_cls==1:
                    row = [image_name[idx], proba[idx][0].cpu()]
                else:
                    row = [image_name[idx], proba[idx][1].cpu().item()]
            
                row_df = pd.DataFrame([row], columns = ['filename', 'pred'])
                results_df = pd.concat([results_df, row_df])
            
            if i%100 == 99:
                print(i)
                

    results_df.to_csv('./result/'+file_name+'.csv', header=True, index=False)
    
        
def submit_probs(model, data_loader, device):
    model.eval()

    probs = []
    image_names = []
    with torch.no_grad():
        for i,data in enumerate(data_loader, 0):
            inputs, image_name = data['image'], data['image_name']
            inputs = inputs.to(device)

            outputs = model.forward(inputs)

            prob = torch.nn.Softmax(dim=1)(outputs)
            probs.append(prob.cpu().numpy())
            image_names.append(image_name)
            
            if i%100 == 99:
                print(i)
        
        probs = np.concatenate(probs, 0)
        image_names = np.concatenate(image_names, 0)
    dic = {'probs' : probs[:, 1], 'names' : image_names}

    return dic
