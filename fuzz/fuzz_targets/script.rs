#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct ScriptInput {
    script: Vec<u8>,
    sig_data: Vec<u8>,
}

fuzz_target!(|input: ScriptInput| {
    // Property 1: find_and_delete should never panic
    let result = script::find_and_delete(&input.script, &input.sig_data);

    // Property 2: result should be no longer than original
    assert!(result.len() <= input.script.len());

    // Note: find_and_delete is NOT idempotent in general — removing a pattern
    // can merge surrounding bytes into a new instance of the pattern.
    // This matches Bitcoin Core's FindAndDelete semantics (single pass).

    // Property 4: push_data should never panic and should produce non-empty output
    if input.sig_data.len() <= 65535 {
        let pushed = script::push_data(&input.sig_data);
        assert!(!pushed.is_empty());
    }

    // Property 5: count_non_push_opcodes should never panic (may return Err for invalid scripts)
    let _ = script::count_non_push_opcodes(&input.script);
    let _ = script::count_non_push_opcodes(&result);
});
