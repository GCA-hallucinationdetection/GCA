import json
import models
from utils import load_progress,save_data,predict_w_log
from concurrent.futures import ThreadPoolExecutor, as_completed

data_path = ''
save_path = 'results.json'

with open(data_path,'r',encoding='utf-8') as f:
    content = f.read()
data = json.loads(content)


def extract_facts(entry):
    user_question_prompt = 'Please write a brief Wikipedia for {}.'
    gpt4_gen_fact_prompt = 'Please list the specific factual propositions included in the {text}. Be complete and do not leave any factual claims out. Provide each claim as a separate sentence in a separate bullet point.'
    entry_facts = predict_w_log(gpt4_gen_fact_prompt.format(text=entry['AI']))
    entry_facts = entry_facts.split('\n-')
    print(entry_facts)
    new_entry = {
        'entity': entry['entity'],
        'user_question': user_question_prompt.format(entry['entity']),
        'AI': entry['AI'],
        'label': entry['label'],
        'facts': entry_facts
    }
    return new_entry


def main():
    model_name = 'QALLMEntailment'

    kwargs = {
        'n_questions': 'three',
        'n_regenerate': 3,
        'n_stochastic_questions': 2,
    }

    model = models.all_models[model_name](**kwargs)

    all_prompts = model.get_all_prompts()

    print('Prompts are:')
    for key ,value in all_prompts.items():
        print(f'{key}: {value}')

    processed_dataset = load_progress(save_path)
    current_processed_data_length = len(processed_dataset)

    def get_single_result(entry):
        new_entry = extract_facts(entry)
        single_result = {}
        uncertainty = 0
        entity, user_question, init_reply, init_reply_label, facts = new_entry.values()
        propositions = facts
        single_result[f'{entity}'] = dict(
            init_reply=init_reply,
            init_reply_label=init_reply_label, facts=facts, uncertainties=[],
            propositions=dict())

        for pidx, proposition in enumerate(propositions):
            text_so_far = ' '.join(propositions[:pidx]) if pidx > 0 else None

            print(f'Currently dealing with proposition {pidx}: {proposition}')

            single_result[f'{entity}']['propositions'][f'prop-{pidx}'] = {}
            uncertainty = model.check_truth(
                rp=single_result[f'{entity}']['propositions'][f'prop-{pidx}'],
                data=dict(
                    user_question=user_question,
                    proposition=proposition,
                    text_so_far=text_so_far)
            )

            print(f'Final uncertainty for proposition {proposition}: {uncertainty}')
            single_result[f'{entity}']['uncertainties'].append(uncertainty)

        print('Final generation with uncertainty')
        for pidx, proposition in enumerate(propositions):
            uncertainties = single_result[f'{entity}']['uncertainties'][pidx]
            print(f'{uncertainties:.3f} {proposition}')
        return single_result, uncertainty

    all_uncertainties = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        entry_futures = {executor.submit(get_single_result, entry): entry
                          for entry in data[current_processed_data_length:]}
        for future in as_completed(entry_futures):
            single_result,uncertainty= future.result()
            all_uncertainties.append(uncertainty)
            processed_dataset.append(single_result)
            save_data(processed_dataset,save_path)


    # results['metrics'] = utils.get_metrics(all_labels, all_uncertainties)
    # out = dict(results=results, export_predictions=model.export_predictions)

if __name__ == '__main__':
    main()


