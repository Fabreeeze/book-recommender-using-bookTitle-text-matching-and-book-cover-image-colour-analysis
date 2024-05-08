// DOM elements
const searchForm = document.getElementById('search-form');
const searchResults = document.getElementById('search-results');

// Form submission handler
searchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Collect form data
    const formData = new FormData(searchForm);
    
    // Send form data to Flask server using Fetch API
    try {
        const response = await fetch('/search', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        // Display search results
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
    }
});

// Function to display search results
function displayResults(data) {
    // Clear previous results
    searchResults.innerHTML = '';
    
    // Iterate over each book in the results
    data.forEach(book => {
        // Create HTML elements to display book details
        const bookElement = document.createElement('div');
        bookElement.classList.add('book', 'bg-white', 'p-4', 'rounded-lg', 'shadow-md');
        bookElement.innerHTML = `
            <a href="${book.productURL}" target="_blank">
                <img src="${book.imgUrl}.jpg" alt="${book.title}" class="w-full h-auto mb-2">
                <h2 class="text-lg font-semibold text-gray-800">${book.title}</h2>
                <p class="text-gray-600">${book.author}</p>
                <div class="flex justify-between mt-2">
                    <p class="text-gray-700">${book.stars} Stars</p>
                    <p class="text-gray-700">${book.price}</p>
                </div>
            </a>
        `;
        
        // Append book element to search results container
        searchResults.appendChild(bookElement);
    });
    
    // Show search results container
    searchResults.classList.remove('hidden');
}
