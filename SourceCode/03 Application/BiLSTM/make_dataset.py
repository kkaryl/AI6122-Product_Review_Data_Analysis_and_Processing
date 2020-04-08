import pandas as pd
import json
import re
def read_data(filename):
    review_list=[]
    with open(filename,'r',encoding='utf-8') as f:
        line = f.readline()
        while(line):
            review_data_dict = json.loads(line)
            line = f.readline()
            review_list.append(review_data_dict)
    return review_list


review_list=read_data('CellPhoneReview.json')
print(len(review_list))
review_text_list=[]
sentiment_score_list=[]
pos_cnt=neg_cnt=neu_cnt=0
for review_item in review_list:
    assert 'overall' in review_item.keys()
    assert 'reviewText' in review_item.keys()
    review_item['reviewText']= re.sub('[!#?,.:";()]', '', review_item['reviewText'])
    if int(review_item['overall']) >3:
      sentiment_score_list.append(2)
      assert 'reviewText' in review_item.keys()
      review_text_list.append(review_item['reviewText'])
      pos_cnt+=1
    elif int(review_item['overall']) <3:
      sentiment_score_list.append(0)
      assert 'reviewText' in review_item.keys()
      review_text_list.append(review_item['reviewText'])
      neg_cnt+=1
    else:
      sentiment_score_list.append(1)
      assert 'reviewText' in review_item.keys()
      review_text_list.append(review_item['reviewText'])
      neu_cnt+=1
assert len(review_text_list)==len(sentiment_score_list)
print(len(review_text_list))
print(pos_cnt)
print(neu_cnt)
print(neg_cnt)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(review_text_list,sentiment_score_list,test_size=0.3,random_state=12)
train_dict={'text':X_train,'label':Y_train}
test_dict={'text':X_test,'label':Y_test}
pd.DataFrame(train_dict).to_csv('train.csv',header=None,index=None)
pd.DataFrame(test_dict).to_csv('test.csv',header=None,index=None)
print(len(X_train))
print(len(X_test))

