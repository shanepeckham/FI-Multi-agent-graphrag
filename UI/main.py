#!/usr/bin/env python3
"""
API Query UI - A Python GUI application for querying team data
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import requests
import json
import threading
import re


class MarkdownRenderer:
    """Simple markdown renderer for tkinter Text widget"""
    
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.setup_tags()
    
    def setup_tags(self):
        """Configure text tags for markdown styling"""
        # Headers with better spacing and colors
        self.text_widget.tag_configure("h1", font=("Arial", 16, "bold"), 
                                     spacing1=10, spacing3=5, foreground="#2c3e50")
        self.text_widget.tag_configure("h2", font=("Arial", 14, "bold"), 
                                     spacing1=8, spacing3=4, foreground="#34495e")
        self.text_widget.tag_configure("h3", font=("Arial", 12, "bold"), 
                                     spacing1=6, spacing3=3, foreground="#566573")
        
        # Text formatting with better visibility
        self.text_widget.tag_configure("bold", font=("Arial", 10, "bold"), foreground="#2c3e50")
        self.text_widget.tag_configure("italic", font=("Arial", 10, "italic"), foreground="#5d6d7e")
        self.text_widget.tag_configure("code", font=("Courier", 9), 
                                     background="#f8f9fa", foreground="#e74c3c", 
                                     relief="solid", borderwidth=1)
        self.text_widget.tag_configure("code_block", font=("Courier", 9), 
                                     background="#f8f9fa", foreground="#2c3e50",
                                     lmargin1=20, lmargin2=20, spacing1=5, spacing3=5,
                                     relief="solid", borderwidth=1)
        
        # Lists with proper indentation
        self.text_widget.tag_configure("bullet", lmargin1=20, lmargin2=40, 
                                     spacing1=2, foreground="#2c3e50")
        self.text_widget.tag_configure("numbered", lmargin1=20, lmargin2=40, 
                                     spacing1=2, foreground="#2c3e50")
        
        # Links with better visibility
        self.text_widget.tag_configure("link", foreground="#3498db", underline=True)
        
        # Default text tag for regular content
        self.text_widget.tag_configure("normal", font=("Arial", 10), 
                                     foreground="#2c3e50", spacing1=2)
    
    def render(self, markdown_text):
        """Render markdown text in the text widget"""
        self.text_widget.delete(1.0, tk.END)
        
        if not markdown_text.strip():
            self.text_widget.insert(tk.END, "No content to display", "normal")
            return
        
        lines = markdown_text.split('\n')
        in_code_block = False
        
        for i, line in enumerate(lines):
            original_line = line
            line = line.rstrip()
            
            # Handle empty lines
            if not line.strip():
                self.text_widget.insert(tk.END, '\n')
                continue
            
            # Code blocks
            if line.startswith('```'):
                in_code_block = not in_code_block
                if not in_code_block:
                    self.text_widget.insert(tk.END, '\n')
                continue
            
            if in_code_block:
                self.text_widget.insert(tk.END, original_line + '\n', "code_block")
                continue
            
            # Headers
            if line.startswith('### '):
                self.text_widget.insert(tk.END, line[4:] + '\n', "h3")
            elif line.startswith('## '):
                self.text_widget.insert(tk.END, line[3:] + '\n', "h2")
            elif line.startswith('# '):
                self.text_widget.insert(tk.END, line[2:] + '\n', "h1")
            
            # Lists
            elif line.startswith('- ') or line.startswith('* '):
                self.text_widget.insert(tk.END, f"• {line[2:]}", "bullet")
                self.render_inline_formatting_in_place(f"• {line[2:]}")
                self.text_widget.insert(tk.END, '\n')
            elif re.match(r'^\d+\. ', line):
                match = re.match(r'^(\d+)\. (.+)', line)
                if match:
                    numbered_text = f"{match.group(1)}. {match.group(2)}"
                    self.text_widget.insert(tk.END, numbered_text, "numbered")
                    self.render_inline_formatting_in_place(numbered_text)
                    self.text_widget.insert(tk.END, '\n')
            
            # Regular text with inline formatting
            else:
                self.render_inline_formatting(line + '\n')
    
    def render_inline_formatting(self, text):
        """Render inline markdown formatting"""
        if not text.strip():
            self.text_widget.insert(tk.END, '\n')
            return
            
        pos = 0
        while pos < len(text):
            # Bold text **text**
            bold_match = re.search(r'\*\*(.*?)\*\*', text[pos:])
            if bold_match:
                start = pos + bold_match.start()
                end = pos + bold_match.end()
                
                # Insert text before bold with normal tag
                if start > pos:
                    self.text_widget.insert(tk.END, text[pos:start], "normal")
                # Insert bold text
                self.text_widget.insert(tk.END, bold_match.group(1), "bold")
                pos = end
                continue
            
            # Italic text *text* (but not **text**)
            italic_match = re.search(r'(?<!\*)\*([^*]+)\*(?!\*)', text[pos:])
            if italic_match:
                start = pos + italic_match.start()
                end = pos + italic_match.end()
                
                # Insert text before italic with normal tag
                if start > pos:
                    self.text_widget.insert(tk.END, text[pos:start], "normal")
                # Insert italic text
                self.text_widget.insert(tk.END, italic_match.group(1), "italic")
                pos = end
                continue
            
            # Inline code `code`
            code_match = re.search(r'`(.*?)`', text[pos:])
            if code_match:
                start = pos + code_match.start()
                end = pos + code_match.end()
                
                # Insert text before code with normal tag
                if start > pos:
                    self.text_widget.insert(tk.END, text[pos:start], "normal")
                # Insert code text
                self.text_widget.insert(tk.END, code_match.group(1), "code")
                pos = end
                continue
            
            # Links [text](url)
            link_match = re.search(r'\[([^\]]+)\]\([^)]+\)', text[pos:])
            if link_match:
                start = pos + link_match.start()
                end = pos + link_match.end()
                
                # Insert text before link with normal tag
                if start > pos:
                    self.text_widget.insert(tk.END, text[pos:start], "normal")
                # Insert link text
                self.text_widget.insert(tk.END, link_match.group(1), "link")
                pos = end
                continue
            
            # No more formatting found, insert rest of text with normal tag
            self.text_widget.insert(tk.END, text[pos:], "normal")
            break
    
    def render_inline_formatting_in_place(self, text):
        """Helper method for rendering inline formatting in list items"""
        # This is a simplified version that just ensures text visibility
        # The actual inline formatting will be handled by the main render method
        pass


class APIQueryUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Team Query Interface")
        self.root.geometry("1000x900")  # Increased from 800x700 to give more space
        
        # Configure style
        style = ttk.Style()
        style.theme_use('aqua' if self.root.tk.call('tk', 'windowingsystem') == 'aqua' else 'clam')
        
        # Request management
        self.current_request = None
        self.request_cancelled = False
        self.request_start_time = None
        
        self.setup_ui()
        
        # Add debug menu for troubleshooting
        self.add_debug_menu()
        
        # Default values
        self.endpoint_var.set("http://0.0.0.0:8000/query_team")
        self.graph_query_type_var.set("local")  # Default to local
        self.search_query_type_var.set("SIMPLE")  # Default to SIMPLE
        
    def setup_ui(self):
        """Setup the user interface"""
        main_frame = ttk.Frame(self.root, padding="15")  # Increased padding
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights - give much more space to the response area
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        # Give very high weight to response row, minimal to others
        main_frame.rowconfigure(0, weight=0)  # Title
        main_frame.rowconfigure(1, weight=0)  # Endpoint
        main_frame.rowconfigure(2, weight=0)  # Bearer Token
        main_frame.rowconfigure(3, weight=0)  # Query
        main_frame.rowconfigure(4, weight=0)  # Method dropdowns
        main_frame.rowconfigure(5, weight=0)  # Checkboxes
        main_frame.rowconfigure(6, weight=0)  # Timeout/Buttons
        main_frame.rowconfigure(7, weight=0)  # Progress
        main_frame.rowconfigure(8, weight=20) # Response - much larger weight
        main_frame.rowconfigure(9, weight=0)  # Status
        
        # Title
        title_label = ttk.Label(main_frame, text="Team Query Interface", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Endpoint URL
        ttk.Label(main_frame, text="API Endpoint:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.endpoint_var = tk.StringVar()
        endpoint_entry = ttk.Entry(main_frame, textvariable=self.endpoint_var, width=50)
        endpoint_entry.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Authorization Token
        ttk.Label(main_frame, text="Bearer Token:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.token_var = tk.StringVar()
        token_entry = ttk.Entry(main_frame, textvariable=self.token_var, width=50, show="*")
        token_entry.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Query input - reduced height to give more space to response
        ttk.Label(main_frame, text="Query:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.query_text = scrolledtext.ScrolledText(main_frame, height=3, width=50)  # Reduced from 4 to 3
        self.query_text.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.query_text.insert(tk.END, "What will the revenue in 2030 be?")
        
        # Graph Query Type
        ttk.Label(main_frame, text="Graph Query Type:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.graph_query_type_var = tk.StringVar()
        graph_query_combo = ttk.Combobox(main_frame, textvariable=self.graph_query_type_var, 
                                        values=["local", "drift", "global"], 
                                        state="readonly", width=15)
        graph_query_combo.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # Search Query Type (next to Graph Query Type)
        ttk.Label(main_frame, text="Search Query Type:", anchor="e").grid(row=4, column=1, sticky=tk.E, pady=5, padx=(0, 200))
        self.search_query_type_var = tk.StringVar()
        search_query_combo = ttk.Combobox(main_frame, textvariable=self.search_query_type_var,
                                         values=["SIMPLE", "SEMANTIC"],
                                         state="readonly", width=12)
        search_query_combo.grid(row=4, column=2, sticky=tk.W, pady=5)
        
        # Boolean checkboxes for use_search, use_graph, use_web
        checkboxes_frame = ttk.LabelFrame(main_frame, text="Options", padding="5")
        checkboxes_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.use_search_var = tk.BooleanVar()
        self.use_search_var.set(True)  # Default to True
        use_search_check = ttk.Checkbutton(checkboxes_frame, text="Use Search", variable=self.use_search_var)
        use_search_check.grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        self.use_graph_var = tk.BooleanVar()
        self.use_graph_var.set(True)  # Default to True
        use_graph_check = ttk.Checkbutton(checkboxes_frame, text="Use Graph", variable=self.use_graph_var)
        use_graph_check.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        self.use_web_var = tk.BooleanVar()
        self.use_web_var.set(False)  # Default to False
        use_web_check = ttk.Checkbutton(checkboxes_frame, text="Use Web", variable=self.use_web_var)
        use_web_check.grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        
        self.use_reasoning_var = tk.BooleanVar()
        self.use_reasoning_var.set(False)  # Default to False
        use_reasoning_check = ttk.Checkbutton(checkboxes_frame, text="Use Reasoning", variable=self.use_reasoning_var)
        use_reasoning_check.grid(row=0, column=3, sticky=tk.W)
        
        # Timeout setting
        timeout_frame = ttk.Frame(main_frame)
        timeout_frame.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        ttk.Label(timeout_frame, text="Timeout (min):").pack(side=tk.LEFT, padx=(0, 5))
        self.timeout_var = tk.StringVar()
        self.timeout_var.set("10")  # Default 10 minutes
        timeout_entry = ttk.Entry(timeout_frame, textvariable=self.timeout_var, width=5)
        timeout_entry.pack(side=tk.LEFT)
        
        # Submit and Cancel buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=6, column=2, sticky=tk.E, pady=5)
        
        # Submit button
        self.submit_btn = ttk.Button(buttons_frame, text="Submit Query", command=self.submit_query)
        self.submit_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Cancel button (initially hidden)
        self.cancel_btn = ttk.Button(buttons_frame, text="Cancel", command=self.cancel_request)
        self.cancel_btn.pack(side=tk.RIGHT, padx=(5, 0))
        self.cancel_btn.pack_forget()  # Hide initially
        
        # Progress bar with status
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 5))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.grid(row=0, column=1, sticky=tk.E)
        
        # Response area - much larger and better configured
        response_frame = ttk.LabelFrame(main_frame, text="Response", padding="5")
        response_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        response_frame.columnconfigure(0, weight=1)
        response_frame.rowconfigure(0, weight=1)
        
        # Response text widget with scrollbar and improved visibility - much larger
        self.response_text = tk.Text(response_frame, wrap=tk.WORD, font=("Arial", 11),
                                   bg="white", fg="#2c3e50", 
                                   selectbackground="#3498db", selectforeground="white",
                                   insertbackground="#2c3e50", relief="sunken", borderwidth=1,
                                   height=25, width=100)  # Set explicit larger size
        scrollbar = ttk.Scrollbar(response_frame, orient="vertical", command=self.response_text.yview)
        self.response_text.configure(yscrollcommand=scrollbar.set)
        
        self.response_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Initialize markdown renderer
        self.markdown_renderer = MarkdownRenderer(self.response_text)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def add_debug_menu(self):
        """Add debug menu for troubleshooting display issues"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        debug_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Debug", menu=debug_menu)
        
        debug_menu.add_command(label="Test Markdown Rendering", command=self.test_markdown_rendering)
        debug_menu.add_command(label="Clear Response Area", command=self.clear_response)
        debug_menu.add_command(label="Show Text Info", command=self.show_text_info)
    
    def test_markdown_rendering(self):
        """Test markdown rendering with sample content"""
        test_content = """# Test Header
This is a **bold** test with *italic* text and `inline code`.

## Subheader
- List item 1
- List item 2 with **bold**
- List item 3 with *italic*

```
Code block test
Multiple lines
```

Regular paragraph text should be clearly visible.
"""
        self.markdown_renderer.render(test_content)
        self.status_var.set("Test markdown rendered")
    
    def clear_response(self):
        """Clear the response area"""
        self.response_text.delete(1.0, tk.END)
        self.status_var.set("Response area cleared")
    
    def show_text_info(self):
        """Show text widget configuration info"""
        info = f"""Text Widget Info:
Background: {self.response_text.cget('bg')}
Foreground: {self.response_text.cget('fg')}
Font: {self.response_text.cget('font')}
Tags configured: {', '.join(self.response_text.tag_names())}
"""
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, info, "normal")
        self.status_var.set("Text widget info displayed")
    
    def submit_query(self):
        """Submit the query to the API"""
        endpoint = self.endpoint_var.get().strip()
        query = self.query_text.get(1.0, tk.END).strip()
        graph_query_type = self.graph_query_type_var.get().strip()
        search_query_type = self.search_query_type_var.get().strip()
        bearer_token = self.token_var.get().strip()
        
        # Get checkbox values
        use_search = self.use_search_var.get()
        use_graph = self.use_graph_var.get()
        use_web = self.use_web_var.get()
        use_reasoning = self.use_reasoning_var.get()
        
        # Get timeout setting
        try:
            timeout_minutes = float(self.timeout_var.get())
            if timeout_minutes <= 0:
                raise ValueError("Timeout must be positive")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid timeout value (in minutes)")
            return
        
        if not endpoint:
            messagebox.showerror("Error", "Please enter an API endpoint")
            return
        
        if not query:
            messagebox.showerror("Error", "Please enter a query")
            return
        
        if not graph_query_type:
            messagebox.showerror("Error", "Please select a graph query type")
            return
            
        if not search_query_type:
            messagebox.showerror("Error", "Please select a search query type")
            return
        
        # Reset cancellation flag
        self.request_cancelled = False
        
        # Show cancel button, hide submit button
        self.submit_btn.pack_forget()
        self.cancel_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Start progress and timer
        self.progress.start(10)
        import time
        self.request_start_time = time.time()
        self.status_var.set(f"Sending request... (timeout: {timeout_minutes} min)")
        self.update_progress_timer()
        
        # Run request in separate thread to avoid blocking UI
        thread = threading.Thread(target=self.make_request, args=(endpoint, query, graph_query_type, search_query_type, use_search, use_graph, use_web, use_reasoning, bearer_token, timeout_minutes))
        thread.daemon = True
        self.current_request = thread
        thread.start()
    
    def make_request(self, endpoint, query, graph_query_type, search_query_type, use_search, use_graph, use_web, use_reasoning, bearer_token, timeout_minutes):
        """Make the API request in a separate thread"""
        try:
            # Check if request was cancelled before starting
            if self.request_cancelled:
                return
            
            # Prepare request data with new structure including boolean fields
            data = {
                "query": query,
                "graph_query_type": graph_query_type,
                "search_query_type": search_query_type,
                "use_search": use_search,
                "use_graph": use_graph,
                "use_web": use_web,
                "use_reasoning": use_reasoning
            }
            
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            # Add authorization header if bearer token is provided
            if bearer_token:
                headers['Authorization'] = f'Bearer {bearer_token}'
            
            # Convert timeout to seconds
            timeout_seconds = timeout_minutes * 60
            
            # Make the request with configurable timeout
            response = requests.post(endpoint, json=data, headers=headers, timeout=timeout_seconds)
            
            # Check if request was cancelled during execution
            if self.request_cancelled:
                return
            
            # Schedule UI update in main thread
            self.root.after(0, self.handle_response, response)
            
        except requests.exceptions.Timeout:
            if not self.request_cancelled:
                self.root.after(0, self.handle_error, f"Request timed out after {timeout_minutes} minutes. The server may still be processing your request.")
        except requests.exceptions.RequestException as e:
            if not self.request_cancelled:
                self.root.after(0, self.handle_error, str(e))
        except Exception as e:
            if not self.request_cancelled:
                self.root.after(0, self.handle_error, f"Unexpected error: {str(e)}")
    
    def handle_response(self, response):
        """Handle the API response"""
        try:
            # Reset UI state
            self.progress.stop()
            self.progress_label.config(text="")
            self.submit_btn.pack(side=tk.RIGHT, padx=(5, 0))
            self.cancel_btn.pack_forget()
            self.current_request = None
            self.request_start_time = None
            
            if response.status_code == 200:
                # Try to parse as JSON first
                try:
                    response_data = response.json()
                    if isinstance(response_data, str):
                        # Response is a string (markdown)
                        self.markdown_renderer.render(response_data)
                    else:
                        # Response is JSON, try to extract markdown content
                        if 'content' in response_data:
                            self.markdown_renderer.render(response_data['content'])
                        elif 'response' in response_data:
                            self.markdown_renderer.render(response_data['response'])
                        elif 'message' in response_data:
                            self.markdown_renderer.render(response_data['message'])
                        elif 'text' in response_data:
                            self.markdown_renderer.render(response_data['text'])
                        else:
                            # Display JSON as formatted text with proper styling
                            json_str = json.dumps(response_data, indent=2)
                            self.response_text.delete(1.0, tk.END)
                            self.response_text.insert(tk.END, json_str, "normal")
                except json.JSONDecodeError:
                    # Response is not JSON, treat as plain text/markdown
                    try:
                        self.markdown_renderer.render(response.text)
                    except Exception:
                        # Fallback to plain text if markdown rendering fails
                        self.response_text.delete(1.0, tk.END)
                        self.response_text.insert(tk.END, f"Response (plain text):\n\n{response.text}", "normal")
                
                self.status_var.set(f"Success - Status: {response.status_code}")
            else:
                # Handle error response with proper styling
                error_msg = f"Error {response.status_code}: {response.text}"
                self.response_text.delete(1.0, tk.END)
                self.response_text.insert(tk.END, error_msg, "normal")
                self.status_var.set(f"Error - Status: {response.status_code}")
                
        except Exception as e:
            self.handle_error(f"Error processing response: {str(e)}")
    
    def handle_error(self, error_message):
        """Handle errors"""
        # Reset UI state
        self.progress.stop()
        self.progress_label.config(text="")
        self.submit_btn.pack(side=tk.RIGHT, padx=(5, 0))
        self.cancel_btn.pack_forget()
        self.current_request = None
        self.request_start_time = None
        
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, f"Error: {error_message}")
        self.status_var.set("Error occurred")
        
        messagebox.showerror("Request Error", error_message)
    
    def cancel_request(self):
        """Cancel the current request"""
        self.request_cancelled = True
        if self.current_request:
            # Note: requests doesn't support cancellation, but we can ignore the response
            pass
        
        # Reset UI
        self.progress.stop()
        self.progress_label.config(text="")
        self.submit_btn.pack(side=tk.RIGHT, padx=(5, 0))
        self.cancel_btn.pack_forget()
        self.status_var.set("Request cancelled")
        
        # Clear response area
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, "Request was cancelled by user.")
    
    def update_progress_timer(self):
        """Update the progress timer display"""
        if hasattr(self, 'request_start_time') and self.request_start_time:
            import time
            elapsed = time.time() - self.request_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            
            if minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"
            
            self.progress_label.config(text=f"Elapsed: {time_str}")
            
            # Schedule next update if request is still running
            if not self.request_cancelled and self.current_request:
                self.root.after(1000, self.update_progress_timer)


def main():
    """Main function to run the application"""
    root = tk.Tk()
    APIQueryUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
