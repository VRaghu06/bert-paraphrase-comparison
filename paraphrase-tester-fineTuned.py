from transformers import pipeline

classifier = pipeline("text-classification" , model = "fine-tuning-trial1" , tokenizer = "fine-tuning-trial1", device=0)

result = classifier([{"text":"The company reported strong earnings." , "text_pair":"The firm posted strong profits."}, {"text":"She did not attend the party." , "text_pair":"The party was not graced with her presence."}])

print(result)