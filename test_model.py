import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def main():
    parser = argparse.ArgumentParser(description='Test pruned model')
    parser.add_argument(
        '--model-path',
        type=str,
        default='artifacts/GigaChat3-10B-A1.8B-bf16/evol-codealpaca-v1/pruned_models/reap-seed_42-0.50',
        help='Path to pruned model',
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='Write a Python function to calculate fibonacci:',
        help='Prompt for generation',
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=1000,
        help='Maximum new tokens to generate',
    )
    args = parser.parse_args()

    print(f'Loading model from {args.model_path}...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    model.generation_config = GenerationConfig.from_pretrained(args.model_path)
    print('Model loaded!')

    messages = [{'role': 'user', 'content': args.prompt}]

    print(f'\\nPrompt: {args.prompt}\\n')
    print('-' * 50)

    input_tensor = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors='pt'
    )
    attention_mask = torch.ones_like(input_tensor)
    outputs = model.generate(
        input_tensor.to(model.device),
        attention_mask=attention_mask.to(model.device),
        max_new_tokens=args.max_new_tokens
    )
    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=False)
    print(result)


if __name__ == '__main__':
    main()
