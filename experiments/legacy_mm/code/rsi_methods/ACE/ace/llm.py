"""
==============================================================================
llm.py
==============================================================================

This file contains the LLM class for the project.

"""
import time
import random
from datetime import datetime

from benchmark.backend import BackendDispatcher
from .logger import log_llm_call, log_problematic_request

def timed_llm_call(
    dispatcher: BackendDispatcher,
    backend_name: str,
    model: str,
    prompt: str,
    role: str,
    call_id: str,
    max_tokens: int = 4096,
    log_dir=None,
    sleep_seconds: float = 15,
    retries_on_timeout: int = 5,
    attempt: int = 1,
    use_json_mode: bool = False,
    reasoning_effort: str | None = None,
    image_paths=None,
):
    """
    Make a timed LLM call with error handling and retry logic.
    
    EMPTY RESPONSE HANDLING STRATEGY:
    - Training calls (call_id starts with 'train_'): Skip the entire training sample
    - Test calls (call_id starts with 'test_'): Mark as incorrect (return wrong answers)
    - All empty responses are logged to problematic_requests/ for SambaNova support analysis
    
    For test calls specifically: Returns "INCORRECT_DUE_TO_EMPTY_RESPONSE" repeated 4 times
    (comma-separated) to handle the 4-question format used in financial NER evaluation.
    
    Args:
        dispatcher: Provider-neutral backend dispatcher
        backend_name: Registered backend name
        model: Model name to use
        prompt: Text prompt to send
        role: Role for logging (generator, reflector, curator)
        call_id: Unique identifier for this call (format: {train|test}_{role}_{details})
        max_tokens: Maximum tokens to generate
        log_dir: Directory for detailed logging
        sleep_seconds: Base sleep time between retries
        retries_on_timeout: Maximum number of retries for timeouts/rate limits/empty responses
        attempt: Current attempt number (for recursive calls)
        use_json_mode: Whether to use JSON mode for structured output
        reasoning_effort: Optional reasoning effort for this request
    
    Returns:
        tuple: (response_text, call_info_dict)
        
    Special return values for empty responses:
        - Training: ("INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, ...", call_info)
        - Testing: ("INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, ...", call_info)
    """
    start_time = time.time()
    prompt_time = time.time()
    
    print(f"[{role.upper()}] Starting call {call_id}...")
    attempt_ledger = []
    
    while True:
        try:
            # Prepare provider-neutral backend parameters
            api_params = {
                "backend_name": backend_name,
                "model": model,
                "prompt": prompt,
                "max_output_tokens": max_tokens,
                "output_format": "json_object" if use_json_mode else None,
                "reasoning_effort": reasoning_effort,
            }
            if image_paths:
                api_params["image_paths"] = list(image_paths)

            call_start = time.time()
            attempt_entry = {
                "attempt": attempt,
                "started_at": datetime.now().isoformat(),
                "usage_complete": False,
                "unknown_may_have_billed": True,
            }
            attempt_ledger.append(attempt_entry)
            response = dispatcher.distribute(**api_params)
            call_end = time.time()

            # Check if response is valid
            if not response or not response.text:
                raise Exception("Empty response from API")

            response_time = time.time()
            total_time = response_time - start_time
            response_content = response.text
            usage = response.usage
            attempt_entry.update(
                status="response_received",
                total_tokens=usage.total_tokens if usage else None,
                usage_complete=bool(usage and isinstance(usage.total_tokens, int)),
                unknown_may_have_billed=not bool(usage and isinstance(usage.total_tokens, int)),
            )

            call_info = {
                "role": role,
                "call_id": call_id,
                "model": response.model or model,
                "provider": response.provider,
                "prompt": prompt,
                "response": response_content,
                "reasoning": response.reasoning,
                "status": response.status,
                "finish_reason": response.finish_reason,
                "prompt_time": prompt_time - start_time,
                "response_time": response_time - prompt_time,
                "total_time": total_time,
                "call_time": call_end - call_start,
                "prompt_length": len(prompt),
                "response_length": len(response_content),
                "prompt_num_tokens": usage.input_tokens if usage else None,
                "response_num_tokens": usage.output_tokens if usage else None,
                "total_num_tokens": usage.total_tokens if usage else None,
                "cached_prompt_num_tokens": (
                    usage.cached_input_tokens if usage else None
                ),
                "reasoning_num_tokens": (
                    usage.reasoning_tokens if usage else None
                ),
                "attempt_ledger": list(attempt_ledger),
                "usage_complete": all(x["usage_complete"] for x in attempt_ledger),
            }
            if not call_info["usage_complete"]:
                call_info["known_successful_response_tokens"] = call_info["total_num_tokens"]
                for key in ("prompt_num_tokens", "response_num_tokens",
                            "total_num_tokens", "cached_prompt_num_tokens",
                            "reasoning_num_tokens"):
                    call_info[key] = None
            if image_paths:
                from benchmark.visual_evidence import image_file
                import hashlib
                call_info["visual_inputs"] = [
                    {"path": str(image_file(p)),
                     "sha256": hashlib.sha256(image_file(p).read_bytes()).hexdigest()}
                    for p in image_paths
                ]
            
            print(f"[{role.upper()}] Call {call_id} completed in {total_time:.2f}s")
            
            if log_dir:
                log_llm_call(log_dir, call_info)
            
            return response_content, call_info
            
        except Exception as e:
            if attempt_ledger:
                attempt_ledger[-1].update(status="failed", error_type=type(e).__name__)
                if log_dir:
                    # Preserve the failed intent even if a later retry succeeds.
                    import json
                    from pathlib import Path
                    ledger_path = Path(log_dir) / "request_attempts.jsonl"
                    ledger_path.parent.mkdir(parents=True, exist_ok=True)
                    with ledger_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "call_id": call_id, "role": role,
                            **attempt_ledger[-1],
                        }) + "\n")
            # Check for both timeout and rate limit errors
            is_timeout = any(k in str(e).lower() for k in ["timeout", "timed out", "connection"])
            is_rate_limit = any(k in str(e).lower() for k in ["rate limit", "429", "rate_limit_exceeded"])
            is_empty_response = "empty response" in str(e).lower() or "api returned none content" in str(e).lower()
            
            # Check for server errors (500, 502, 503, etc.) that should be retried
            is_server_error = False
            if hasattr(e, 'response'):
                try:
                    status_code = getattr(e.response, 'status_code', None)
                    if status_code and status_code >= 500:
                        is_server_error = True
                        print(f"[{role.upper()}] Server error detected: HTTP {status_code}")
                except:
                    pass
            
            # Also check for 500 errors in the error message itself
            if any(k in str(e).lower() for k in ["500 internal server error", "internal server error", "502 bad gateway", "503 service unavailable"]):
                is_server_error = True
                print(f"[{role.upper()}] Server error detected in message: {str(e)[:100]}...")
            
            # Debug empty response issues
            if is_empty_response:
                print(f"\n🚨 DEBUG: Empty response detected for {call_id}")
                print(f"📝 Exception type: {type(e).__name__}")
                print(f"📝 Exception message: {str(e)}")
                print(f"📝 Using JSON mode: {use_json_mode}")
                print(f"📝 Model: {model}")
                print(f"📝 Prompt length: {len(prompt)}")
                print(f"📝 Prompt preview (first 500 chars):")
                print(f"    {prompt[:500]}...")
                print(f"📝 Full exception details: {repr(e)}")
                if hasattr(e, 'response'):
                    print(f"📝 Raw response object: {e.response}")
                    if hasattr(e.response, 'text'):
                        print(f"📝 Raw response text: {e.response.text}")
                    if hasattr(e.response, 'content'):
                        print(f"📝 Raw response content: {e.response.content}")
                print("-" * 60)
                
                # Log problematic requests for SambaNova support
                log_problematic_request(call_id, prompt, model, api_params, e, log_dir, False,
                                       None)
            
            # For empty responses, we handle differently based on context
            if is_empty_response:
                # Log the problematic request for SambaNova support
                log_problematic_request(call_id, prompt, model, api_params, e, log_dir, False,
                                       None)
                
                # Check if this is a training or test call to decide behavior
                if call_id.startswith('train_'):
                    # In training: Mark as incorrect answer (same as testing)
                    print(f"[{role.upper()}] 🚨 Empty response in training - marking as INCORRECT for {call_id}")
                    error_time = time.time()
                    call_info = {
                        "role": role,
                        "call_id": call_id,
                        "model": model,
                        "prompt": prompt,
                        "error": "TRAINING_INCORRECT: " + str(e),
                        "total_time": error_time - start_time,
                        "prompt_length": len(prompt),
                        "response_length": 0,
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                        "datetime": datetime.now().isoformat(),
                        "training_marked_incorrect_due_to_empty_response": True
                    }
                    if log_dir:
                        log_llm_call(log_dir, call_info)
                    
                    # Return a response that will be marked as incorrect
                    # For the 4-question format, we return 4 wrong answers
                    incorrect_response = "INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE"
                    return incorrect_response, call_info
                
                elif call_id.startswith('test_'):
                    # In testing: Treat as incorrect answer
                    print(f"[{role.upper()}] 🚨 Empty response in testing - marking as INCORRECT for {call_id}")
                    error_time = time.time()
                    call_info = {
                        "role": role,
                        "call_id": call_id,
                        "model": model,
                        "prompt": prompt,
                        "error": "TEST_INCORRECT: " + str(e),
                        "total_time": error_time - start_time,
                        "prompt_length": len(prompt),
                        "response_length": 0,
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                        "datetime": datetime.now().isoformat(),
                        "test_marked_incorrect_due_to_empty_response": True
                    }
                    if log_dir:
                        log_llm_call(log_dir, call_info)
                    
                    # Return a response that will be marked as incorrect
                    # For the 4-question format, we return 4 wrong answers
                    incorrect_response = "INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE"
                    return incorrect_response, call_info
            
            # Retry logic for timeouts, rate limits, and server errors
            if (is_timeout or is_rate_limit or is_server_error) and attempt < retries_on_timeout:
                attempt += 1
                if is_rate_limit:
                    error_type = "rate limited"
                    base_sleep = sleep_seconds * 2
                elif is_server_error:
                    error_type = "server error (500+)"
                    base_sleep = sleep_seconds * 1.5  # Moderate delay for server errors
                elif is_empty_response:
                    error_type = "returned empty response"
                    base_sleep = sleep_seconds
                else:
                    error_type = "timed out"
                    base_sleep = sleep_seconds
                jitter = random.uniform(0.5, 1.5)  # Add jitter to avoid thundering herd
                sleep_time = base_sleep * jitter
                print(f"[{role.upper()}] Call {call_id} {error_type}, sleeping {sleep_time:.1f}s then retrying "
                      f"({attempt}/{retries_on_timeout})...")
                time.sleep(sleep_time)
                continue
            
            error_time = time.time()
            call_info = {
                "role": role,
                "call_id": call_id,
                "model": model,
                "prompt": prompt,
                "error": str(e),
                "total_time": error_time - start_time,
                "prompt_length": len(prompt),
                "attempt": attempt,
            }
            
            print(f"[{role.upper()}] Call {call_id} failed after {error_time - start_time:.2f}s: {e}")
            
            if log_dir:
                log_llm_call(log_dir, call_info)
            
            raise e
