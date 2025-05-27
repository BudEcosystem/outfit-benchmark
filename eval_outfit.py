import json
import requests
import base64
import re
from typing import List, Dict

from langchain_community.chat_models import ChatLiteLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

LLM_MODEL_QWEN_VL = "Qwen/Qwen2.5-VL-7B-Instruct"
LLM_BASE_QWEN_VL = ""

LLM_TEMPERATURE = 0.2
API_KEY = "YOUR_API_KEY"

# Initialize the vision-language model
chat_llm_vl = ChatLiteLLM(
    model=LLM_MODEL_QWEN_VL,
    temperature=LLM_TEMPERATURE,
    api_base=LLM_BASE_QWEN_VL,
    api_key=API_KEY,
)

def _download_to_b64(url: str, timeout: int = 10) -> str:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("utf-8")


def _prepare_images_block(image_urls: List[str]) -> List[Dict]:
    block = []
    for url in image_urls:
        b64 = _download_to_b64(url)
        block.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return block


def _invoke_qwen_vl(prompt_blocks: List[Dict]) -> str:
    human_msg = HumanMessage(content=prompt_blocks)
    chat_prompt = ChatPromptTemplate.from_messages([human_msg])
    resp = chat_llm_vl.invoke(chat_prompt.format_messages()).content.strip()
    return resp


def _parse_score_and_reason(resp: str) -> (float, str):
    """
    Extracts the first numeric score and optional reason after the first semicolon.
    """
    match = re.search(r"([0-9]*\.?[0-9]+)", resp)
    if not match:
        raise ValueError(f"No numeric score found in response: '{resp}'")
    score = float(match.group(1))
    reason = ""
    if ";" in resp:
        _, rest = resp.split(";", 1)
        reason = rest.strip()
        if reason.lower().startswith("reason:"):
            reason = reason[len("reason:"):].strip()
    return score, reason


def rate_visual_similarity(
    images_a: List[str], images_b: List[str]
) -> Dict[str, any]:
    blocks = _prepare_images_block(images_a) + _prepare_images_block(images_b)
    blocks.append({
        "type": "text",
        "text": (
            "You are given two groups of fashion product images. "
            "The first set is Input Outfit and the second is Generated Outfit. "
            "Assess their visual similarity. Respond in one line as: "
            "similarity: <score 0-1>; reason: <brief rationale>."
        ),
    })
    resp = _invoke_qwen_vl(blocks)
    score, reason = _parse_score_and_reason(resp)
    return {"similarity": score, "reason": reason}


def rate_closeness_to_query(
    images: List[str], description: str, query: str
) -> Dict[str, any]:
    blocks = _prepare_images_block(images)
    blocks.append({
        "type": "text",
        "text": (
            f"User query: '{query}'.\n"
            f"Outfit description: '{description}'.\n"
            "Assess how close this outfit is to the query. "
            "Respond in one line as: closeness: <score 0-1>; reason: <brief rationale>."
        ),
    })
    resp = _invoke_qwen_vl(blocks)
    score, reason = _parse_score_and_reason(resp)
    return {"closeness": score, "reason": reason}


def rate_relevance(
    images: List[str], description: str, query: str
) -> Dict[str, any]:
    blocks = _prepare_images_block(images)
    blocks.append({
        "type": "text",
        "text": (
            f"User query: '{query}'.\n"
            f"Outfit description: '{description}'.\n"
            "Assess the relevance of this outfit to the query. "
            "Respond in one line as: relevance: <score 0-1>; reason: <brief rationale>."
        ),
    })
    resp = _invoke_qwen_vl(blocks)
    score, reason = _parse_score_and_reason(resp)
    return {"relevance": score, "reason": reason}


def rate_helpfulness(
    images: List[str], description: str, query: str
) -> Dict[str, any]:
    blocks = _prepare_images_block(images)
    blocks.append({
        "type": "text",
        "text": (
            f"User query: '{query}'.\n"
            f"Outfit description: '{description}'.\n"
            "Assess how helpful this outfit is in addressing the user's needs. "
            "Respond in one line as: helpfulness: <score 0-1>; reason: <brief rationale>."
        ),
    })
    resp = _invoke_qwen_vl(blocks)
    score, reason = _parse_score_and_reason(resp)
    return {"helpfulness": score, "reason": reason}


def rate_quality_of_outfit(
    images: List[str]
) -> Dict[str, any]:
    blocks = _prepare_images_block(images)
    blocks.append({
        "type": "text",
        "text": (
            "You are given images of a single fashion outfit. "
            "Assess the overall quality of the outfit on a scale 0-1. "
            "Respond in one line as: quality: <score 0-1>; reason: <brief rationale>."
        ),
    })
    resp = _invoke_qwen_vl(blocks)
    score, reason = _parse_score_and_reason(resp)
    return {"quality": score, "reason": reason}


def main(
    json_path: str,
    output_path: str = "rated_outfits_results.json"
):
    with open(json_path, "r") as f:
        data = json.load(f)

    results = {}

    for query, section in data.items():
        input_outfits = section.get("input_outfits", [])
        gen_outfits = section.get("generated_outfits", [])
        section_results = []

        for inp in input_outfits:
            inp_desc = inp.get("collection_name", "")
            inp_urls = [p.get("image_url") for p in inp.get("products", []) if p.get("image_url")]

            # Input metrics
            closeness_inp = rate_closeness_to_query(inp_urls, inp_desc, query)
            relevance_inp = rate_relevance(inp_urls, inp_desc, query)
            helpfulness_inp = rate_helpfulness(inp_urls, inp_desc, query)
            quality_inp = rate_quality_of_outfit(inp_urls)

            for gen in gen_outfits:
                gen_desc = gen.get("description", "")
                gen_urls = [p.get("primary_image_url") for p in gen.get("products", []) if p.get("primary_image_url")]

                # Generated metrics
                sim = rate_visual_similarity(inp_urls, gen_urls)
                closeness_gen = rate_closeness_to_query(gen_urls, gen_desc, query)
                relevance_gen = rate_relevance(gen_urls, gen_desc, query)
                helpfulness_gen = rate_helpfulness(gen_urls, gen_desc, query)
                quality_gen = rate_quality_of_outfit(gen_urls)

                section_results.append({
                    "input_collection": inp.get("collection_name"),
                    "generated_outfit_id": gen.get("outfit_id"),
                    **sim,
                    "closeness_input": closeness_inp,
                    "relevance_input": relevance_inp,
                    "helpfulness_input": helpfulness_inp,
                    "quality_input": quality_inp,
                    "closeness_generated": closeness_gen,
                    "relevance_generated": relevance_gen,
                    "helpfulness_generated": helpfulness_gen,
                    "quality_generated": quality_gen,
                })
        print(section_results)
        results[query] = section_results

    with open(output_path, "w") as out_f:
        json.dump(results, out_f, indent=2)
    print(f"Results written to {output_path}")

if __name__ == "__main__":
    main("combined_outfit_results.json")
