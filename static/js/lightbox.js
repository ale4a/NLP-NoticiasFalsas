const imagenes = document.querySelectorAll('.img-galeria');
const imagenesLight = document.querySelector('.agregar-imagen');
const contenedorLight = document.querySelector('.imagen-light');
const hambuger1 = document.querySelector('.hamburger');

imagenes.forEach(imagen =>{
    imagen.addEventListener('click', ()=>{
        console.log(imagen.getAttribute('src') );
        // necesitaria pasar los datos pero da un error, con la funcion aparecerImagen
        //aparecerImagen(imagen.getAttribute('src'));
    })
})

contenedorLight.addEventListener('click',(e)=>{
    if(e.target!= imagenesLight){
        contenedorLight.classList.toggle('show');
        imagenesLight.classList.toggle('showImage');
        hambuger1.style.opacity = '1'
    }
})

const aparecerImagen = (imagen)=>{
    
    imagenesLight.src = imagen;
    contenedorLight.classList.toggle('show');
    imagenesLight.classList.toggle('showImage');
    hambuger1.style.opacity = '0'
}