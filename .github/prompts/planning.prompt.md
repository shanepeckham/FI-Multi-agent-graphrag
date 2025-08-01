---
mode: 'agent'
---
# Copilot Planning Guidelines

## Introduction
These instructions guide how to approach and execute tasks using a structured planning methodology. Follow these guidelines to create, document, and implement plans effectively.

## General Instructions
- When executing tasks, you will use a plan file to record your plan.
- Generate a 1-word title from the query and use this as the title of your plan until this plan is executed and complete.
- The plan file should be called 'llm-plan-{title}.md' in the '.copilot' directory.
  - Example: For a task about authentication, use 'llm-plan-authentication.md'
- Ensure this plan file is created in the '.copilot' directory.
- Use this file to store the steps of the plan and its progress.
- Make sure you save the steps and progress of steps in the file as you go.
- If the '.copilot' directory doesn't exist, create it first.

## Plan Structure Guidelines
- When creating a plan, organize it into numbered phases (e.g., "Phase 1: Setup Dependencies")
- Break down each phase into specific tasks with numeric identifiers (e.g., "Task 1.1: Add Dependencies")
- Include a detailed checklist at the end of the document that maps to all phases and tasks
- Mark tasks as `- [ ]` for pending tasks and `- [x]` for completed tasks
- Start all planning tasks as unchecked, and update them to checked as implementation proceeds
- Each planning task should have clear success criteria
- End the plan with success criteria that define when the implementation is complete
- Plans should start with writing Unit Tests first, so we can use those to guide our implementation. Same for UI tests when it makes sense.

### Example Plan Structure:
```
# Plan: Authentication

## Phase 1: Test Design
- Task 1.1: Create authentication test cases
- Task 1.2: Setup test environment

## Phase 2: Implementation
- Task 2.1: Implement login functionality
- Task 2.2: Add security features

## Checklist
- [ ] Task 1.1: Create authentication test cases
- [ ] Task 1.2: Setup test environment
- [ ] Task 2.1: Implement login functionality
- [ ] Task 2.2: Add security features

## Success Criteria
- All tests pass
- Authentication flow works end-to-end
```

## Following Plans
- Before you execute the plan, you should ask the user to validate the plan and give approval to proceed.
- Ensure the plan is saved into the '.copilot' directory.
- If the user provides approval, then proceed to implement the plan.
- When coding you need to follow the plan phases and check off the tasks as they are completed.
- As you complete a task, update the plan and mark that task complete before you begin the next task.
- Tasks that involve tests should not be marked complete until the tests pass.
- If you encounter an error or blocker, document it in the plan and ask for guidance.

## Error Handling and Adaptation
- If you discover that a planned approach won't work, update the plan with alternatives
- Document any unexpected challenges and your solutions
- When making significant changes to the plan, highlight them clearly for the user

## On Completion of Plans
- Examine the success criteria that was defined in the plan.
- Assess whether the success criteria has been met.
  - If the success criteria has been met, then terminate.
  - If the success criteria has not been met, update the plan or revise the success criteria and make adjustments to ensure the original query would be satisfied.
  - Only rewrite the plan once if required. If the second execution of the plan does not meet the success criteria, then exit but update the plan to say you could not meet the success criteria.

## Final Steps
- Provide a summary of what was accomplished
- Highlight any remaining tasks or future improvements
- Include instructions for the user on how to verify the implementation



