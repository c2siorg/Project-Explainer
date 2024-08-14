from prompts import *
import pandas as pd
import random
from tqdm.auto import tqdm
import requests



def get_response(prompt: str,url: str = "http://localhost:11434/api/generate",model: str= "llama3"):

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    resp=response.json()
    return resp['response']




def qa_generator(documents: object,
                 N_GENERATIONS: int= 20,):

    print(f"Generating {N_GENERATIONS} QA couples...")

    outputs = []
    for sampled_context in tqdm(random.sample(documents, N_GENERATIONS)):
        # Generate QA couple
        output_QA_couple = get_response (QA_generation_prompt.format(context=sampled_context.text))
        try:
            question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
            answer = output_QA_couple.split("Answer: ")[-1]
            assert len(answer) < 300, "Answer is too long"
            outputs.append(
                {
                    "context": sampled_context.text,
                    "question": question,
                    "answer": answer,
                    "source_doc": sampled_context.metadata,
                }
            )
        except:
            continue
    df=pd.DataFrame(outputs)
    df.to_csv("QA.csv")
    return outputs

def critique_qa(outputs: list,):
    print("Generating critique for each QA couple...")
    for output in tqdm(outputs):
        evaluations = {
            "groundedness": get_response(
                question_groundedness_critique_prompt.format(context=output["context"], question=output["question"]),
            ),
            "relevance": get_response(
                question_relevance_critique_prompt.format(question=output["question"]),
            ),
            "standalone": get_response(
                question_standalone_critique_prompt.format(question=output["question"]),
            ),
        }
        try:
            for criterion, evaluation in evaluations.items():
                score, eval = (
                    int(evaluation.split("Total rating: ")[-1].strip()),
                    evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                )
                output.update(
                    {
                        f"{criterion}_score": score,
                        f"{criterion}_eval": eval,
                    }
                )
        except Exception as e:
            continue
        generated_questions = pd.DataFrame.from_dict(outputs)
        generated_questions = generated_questions.loc[
            (generated_questions["groundedness_score"] >= 4)
            & (generated_questions["relevance_score"] >= 4)
            & (generated_questions["standalone_score"] >= 4)
            ]
        generated_questions.to_csv("generated_questions.csv")
        return outputs