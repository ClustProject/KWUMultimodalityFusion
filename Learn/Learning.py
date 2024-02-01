import torch
from .Metric import accuracy
from .Helper import disc_rank, edge_rank, drop_features2, drop_edges2
from sklearn.metrics import confusion_matrix

def train(train_loader, model, optimizer, epochs, l2_coefficient, batch_size):
    for epoch in range(1, epochs + 1):
        train_acc, train_loss, val_acc, val_loss, count = 0.,0.,0.,0.,0.
        for batch_idx, data in enumerate(train_loader):
            if len(data.y) == batch_size:  # dataset size가 batch size로 안나눠지면 버림
                count += 1.
                optimizer.zero_grad()

                model.train()
                logits = model(data)
                l2_loss = model.l2_regularization(l2_coefficient)
                loss = torch.nn.functional.cross_entropy(logits, data.y) + l2_loss

                loss.backward()
                optimizer.step()

                pred = logits.max(1, keepdim=True)[1]
                train_acc += pred.eq(data.y.view_as(pred)).sum().item()
                train_loss += loss
        average_train_acc = 100. * train_acc / (batch_size*count)
        average_train_loss = train_loss / (batch_size*count)

        if epoch % 10 == 0:
            print("Epoch {} - Train Acc : {}    Train Loss : {}".format(epoch, round(
                average_train_acc.item(), 2), round(average_train_loss.item(), 2)))


def test(test_loader, model, batch_size):
    model.eval()

    count, test_loss, test_acc = 0, 0, 0
    with torch.no_grad():
        for data in test_loader:
            if len(data.y) == batch_size:
                count += 1
                logits = model(data)
                loss = logits.item()
                test_loss += loss

                pred = logits.max(1, keepdim=True)[1]
                test_acc += pred.eq(data.y.view_as(pred)).sum().item()
        average_test_acc = 100. * test_acc / (batch_size * count)
        average_test_loss = test_loss / (batch_size * count)

    return average_test_acc, average_test_loss


def GCA_train(model, optimizer, feature, orig_adj, label, train_identifier, test_identifier, args, isdeap=False):
    #     save_path = args.model_save_path+'subject_dependent/'+date+'/'+sub_idx+'.pt'
    #     early_stopping = EarlyStopping(patience = args.patience, verbose = False, path=save_path)
    best_acc = 0
    best_epoch = 0
    best_model = None
    best_z = None
    #     w = 0.5

    rank = disc_rank(feature, label, train_identifier, args.out_channels)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        x1 = drop_features2(feature, rank, p=args.pf1, threshold=args.tpf1)
        x2 = drop_features2(feature, rank, p=args.pf2, threshold=args.tpf2)
        e1 = drop_edges(orig_adj, p=args.pe1, threshold=args.tpe1)
        e2 = drop_edges(orig_adj, p=args.pe2, threshold=args.tpe2)

        #         x1 = drop_features(feature, adj, p = 0.1, threshold = args.tpf1)
        #         x2 = drop_features(feature, adj, p = 0.2, threshold = args.tpf2)
        #         e1 = drop_edges(adj, p = 0.1, threshold = args.tpe1)
        #         e2 = drop_edges(adj, p = 0.2, threshold = args.tpe2)

        z1 = model(x1, e1)  # ,bias = True)
        #         z1 = model(feature,adj)
        z1 = model.projection(z1)
        z2 = model(x2, e2)
        z2 = model.projection(z2)

        #         ne1 = model.decoder(z1)
        #         ne2 = model.decoder(z2)

        #         ne1 = (ne1-ne1.min())/(ne1.max()-ne1.min())
        #         ne2 = (ne2-ne2.min())/(ne2.max()-ne2.min())
        #         nadj1 = w*adj + (1.-w)*ne1
        #         nadj2 = w*adj + (1.-w)*ne2
        #         nadj = 0.5*(nadj1+nadj2)
        #         print(nadj)

        r1 = model.classification(z1)
        r1_pred = r1[train_identifier]
        r1_y = label[train_identifier]
        # L2 regularization is not implemented yet
        labeled_loss1 = torch.nn.functional.cross_entropy(r1_pred, r1_y)
        r1_acc = accuracy(r1_pred, r1_y, isdeap)

        r2 = model.classification(z2)
        r2_pred = r2[train_identifier]
        r2_y = label[train_identifier]
        # L2 regularization is not implemented yet
        labeled_loss2 = torch.nn.functional.cross_entropy(r2_pred, r2_y)
        r2_acc = accuracy(r2_pred, r2_y, isdeap)

        contrastive_loss = model.loss(z1, z2)
        #         print(contrastive_loss)
        loss = (labeled_loss1 + labeled_loss2) / 2. + contrastive_loss * args.loss_lambda
        #         loss = labeled_loss1 + contrastive_loss*args.loss_lambda

        loss.backward()
        optimizer.step()

        #         orig_adj = nadj.detach().clone().to(device)
        #         print(orig_adj)
        #         adj = nadj.detach().clone().cuda()
        acc = (r1_acc + r2_acc) / 2.
        #         acc = r1_acc

        tr1_pred = r1[test_identifier]
        tr1_y = label[test_identifier]
        tr1_loss = torch.nn.functional.cross_entropy(tr1_pred, tr1_y)
        tr1_acc = accuracy(tr1_pred, tr1_y, isdeap)

        tr2_pred = r2[test_identifier]
        tr2_y = label[test_identifier]
        tr2_acc = accuracy(tr2_pred, tr2_y, isdeap)
        tr2_loss = torch.nn.functional.cross_entropy(tr2_pred, tr2_y)

        #         tr_acc = (tr1_acc + tr2_acc)/2.
        if tr1_acc > tr2_acc:
            result = r1
            tr_acc = tr1_acc
        else:
            result = r2
            tr_acc = tr2_acc

        tr_loss = (tr1_loss + tr2_loss) / 2.
        total_acc = (tr_acc + acc) / 2.

        if tr_acc > best_acc:
            best_acc = tr_acc
            best_epoch = epoch
            best_model = model

            best_result = result
            best_z = z1 if tr1_acc > tr2_acc else z2

        if epoch % 10 == 0:
            print(
                "Epoch {} - Train Acc : {}    Train Loss : {},    Test Acc : {},    Test Loss :{},    Total Acc : {}".format(
                    epoch, round(acc.item(), 2), round(loss.item(), 2), round(tr_acc.item(), 2),
                    round(tr_loss.item(), 2), round(total_acc.item(), 2)))

    #         early_stopping(vloss, model)
    #         if early_stopping.early_stop:
    #             print('Epoch : {} - Ealry Stopping'.format(epoch))
    #             break
    #     model.load_state_dict(torch.load(save_path))
    return model, best_acc, best_epoch, best_model, best_z, best_result


def GCA_train2(model, optimizer, feature, orig_adj, label, train_identifier, test_identifier, args, device, date=None,
               sub_idx=None, isdeap=False):
    #     save_path = args.model_save_path+'subject_dependent/'+date+'/'+sub_idx+'.pt'
    #     early_stopping = EarlyStopping(patience = args.patience, verbose = False, path=save_path)
    best_acc = 0
    best_epoch = 0
    best_model = None
    best_z = None
    #     w = 0.5

    rankf = disc_rank(feature, label, train_identifier, args.out_channels)
    rankf1 = rankf * args.pf1
    rankf2 = rankf * args.pf2
    ranke = edge_rank(orig_adj)
    ranke1 = ranke * args.pe1
    ranke2 = ranke * args.pe2

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        x1 = drop_features2(rankf1, feature, threshold=args.tpf1)
        x2 = drop_features2(rankf2, feature, threshold=args.tpf2)
        e1 = drop_edges2(ranke1, orig_adj, threshold=args.tpe1)
        e2 = drop_edges2(ranke2, orig_adj, threshold=args.tpe2)

        #         x1 = drop_features(feature, adj, p = 0.1, threshold = args.tpf1)
        #         x2 = drop_features(feature, adj, p = 0.2, threshold = args.tpf2)
        #         e1 = drop_edges(adj, p = 0.1, threshold = args.tpe1)
        #         e2 = drop_edges(adj, p = 0.2, threshold = args.tpe2)

        z1 = model(x1, e1)  # ,bias = True)
        #         z1 = model(feature,adj)
        z1 = model.projection(z1)
        z2 = model(x2, e2)
        z2 = model.projection(z2)

        #         ne1 = model.decoder(z1)
        #         ne2 = model.decoder(z2)

        #         ne1 = (ne1-ne1.min())/(ne1.max()-ne1.min())
        #         ne2 = (ne2-ne2.min())/(ne2.max()-ne2.min())
        #         nadj1 = w*adj + (1.-w)*ne1
        #         nadj2 = w*adj + (1.-w)*ne2
        #         nadj = 0.5*(nadj1+nadj2)
        #         print(nadj)

        r1 = model.classification(z1)
        r1_pred = r1[train_identifier]
        r1_y = label[train_identifier]
        # L2 regularization is not implemented yet
        labeled_loss1 = torch.nn.functional.cross_entropy(r1_pred, r1_y)
        r1_acc = accuracy(r1_pred, r1_y, isdeap)

        r2 = model.classification(z2)
        r2_pred = r2[train_identifier]
        r2_y = label[train_identifier]
        # L2 regularization is not implemented yet
        labeled_loss2 = torch.nn.functional.cross_entropy(r2_pred, r2_y)
        r2_acc = accuracy(r2_pred, r2_y, isdeap)

        contrastive_loss = model.loss(z1, z2)
        #         print(contrastive_loss)
        loss = (labeled_loss1 + labeled_loss2) / 2. + contrastive_loss * args.loss_lambda
        #         loss = labeled_loss1 + contrastive_loss*args.loss_lambda

        loss.backward()
        optimizer.step()

        #         orig_adj = nadj.detach().clone().to(device)
        #         print(orig_adj)
        #         adj = nadj.detach().clone().cuda()
        acc = (r1_acc + r2_acc) / 2.
        #         acc = r1_acc

        tr1_pred = r1[test_identifier]
        tr1_y = label[test_identifier]
        tr1_loss = torch.nn.functional.cross_entropy(tr1_pred, tr1_y)
        tr1_acc = accuracy(tr1_pred, tr1_y, isdeap)

        tr2_pred = r2[test_identifier]
        tr2_y = label[test_identifier]
        tr2_acc = accuracy(tr2_pred, tr2_y, isdeap)
        tr2_loss = torch.nn.functional.cross_entropy(tr2_pred, tr2_y)

        #         tr_acc = (tr1_acc + tr2_acc)/2.
        if tr1_acc > tr2_acc:
            result = r1
            tr_acc = tr1_acc
        else:
            result = r2
            tr_acc = tr2_acc

        tr_loss = (tr1_loss + tr2_loss) / 2.
        total_acc = (tr_acc + acc) / 2.

        if tr_acc > best_acc:
            if tr1_acc > tr2_acc:
                best_pred = tr1_pred
            else:
                best_pred = tr2_pred
            best_acc = tr_acc
            best_epoch = epoch
            best_model = model

            best_result = result
            best_z = z1 if tr1_acc > tr2_acc else z2

    cfm = confusion_matrix(tr1_y.cpu().numpy(), best_pred.max(1, keepdim=True)[1].detach().cpu().numpy(),
                           normalize='true')
    #         if epoch % 10 == 0:
    #             print("Epoch {} - Train Acc : {}    Train Loss : {},    Test Acc : {},    Test Loss :{},    Total Acc : {}".format(epoch, round(acc.item(), 2), round(loss.item(),2), round(tr_acc.item(),2), round(tr_loss.item(),2), round(total_acc.item(), 2)))

    #         early_stopping(vloss, model)
    #         if early_stopping.early_stop:
    #             print('Epoch : {} - Ealry Stopping'.format(epoch))
    #             break
    #     model.load_state_dict(torch.load(save_path))
    return model, best_acc, best_epoch, best_model, best_z, best_result, cfm

def GTN_train(feature, adj, label, train_identifier, test_identifier, model, classifier, optimizer, epochs):
    best_acc = 0
    best_epoch = 0
    best_model = None
    best_z = None

    for epoch in range(1, epochs + 1):
        model.train()

        optimizer.zero_grad()
        out = model(feature, adj)
        result = classifier(out)

        train_pred = result[train_identifier]
        train_y = label[train_identifier]
        train_loss = torch.nn.functional.cross_entropy(train_pred, train_y)
        train_acc = accuracy(train_pred, train_y)

        train_loss.backward()
        optimizer.step()

        test_pred = result[test_identifier]
        test_y = label[test_identifier]
        test_loss = torch.nn.functional.cross_entropy(test_pred, test_y)
        test_acc = accuracy(test_pred, test_y)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_model = model
            best_embedding = out

        if epoch % 10 == 0:
            print("Epoch {} - Train Acc : {}    Train Loss : {},    Test Acc : {},    Test Loss :{}".format(epoch,round(train_acc.item(), 2),
                                                                                                            round(train_loss.item(),2),
                                                                                                            round(test_acc.item(),2),
                                                                                                            round(test_loss.item(),2)))

    return model, best_acc, best_epoch, best_model, best_embedding, result

