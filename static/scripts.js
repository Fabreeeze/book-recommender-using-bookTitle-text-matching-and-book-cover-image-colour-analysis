// DOM elements
const searchForm = document.getElementById('search-form');
const searchResults = document.getElementById('search-results');
let genre;
// Form submission handler
searchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Collect form data
    const formData = new FormData(searchForm);
    genre =formData.get('genre');
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
    
    // Separate books by genre
    const sameGenreBooks = [];
    const otherGenreBooks = [];
    
    data.forEach(book => {
        console.log(genre)
        console.log("b=",book.category_name);
        if (book.category_name === genre) {
            sameGenreBooks.push(book);
        } else {
            otherGenreBooks.push(book);
        }
    });
    console.log("same l =",sameGenreBooks.length);
    console.log("diff l =",otherGenreBooks.length);
    // Display same genre books
    if (sameGenreBooks.length > 0) {
        console.log("same genre code")
        const sameGenreHeading = document.createElement('h2');
        sameGenreHeading.textContent = 'Recommended books';
        sameGenreHeading.classList.add('book-heading','col-span-4');
        searchResults.appendChild(sameGenreHeading);
        
        sameGenreBooks.forEach(book => {
            displayBook(book);
        });
    }
    
    // Display separation
    if (sameGenreBooks.length > 0 && otherGenreBooks.length > 0) {
        const separation = document.createElement('hr');
        separation.classList.add('separator','col-span-4');
        searchResults.appendChild(separation);
    }
    
    // Display other genre books
    if (otherGenreBooks.length > 0) {
        const otherGenreHeading = document.createElement('h2');
        otherGenreHeading.textContent = 'Other genre recommendations';
        otherGenreHeading.classList.add('book-heading','col-span-4');
        searchResults.appendChild(otherGenreHeading);
        
        otherGenreBooks.forEach(book => {
            displayBook(book);
        });
    }
    
    // Show search results container
    searchResults.classList.remove('hidden');
}

// Function to display a single book
function displayBook(book) {
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
}
