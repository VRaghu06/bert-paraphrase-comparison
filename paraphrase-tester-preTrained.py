from transformers import pipeline

classifier = pipeline("text-classification" , model = "bert-base-uncased" , device=0)

result = classifier([{"text":"The company reported strong earnings." , "text_pair":"The firm posted strong profits."}, {"text":"She did not attend the party." , "text_pair":"The party was not graced with her presence."}])

print(result)