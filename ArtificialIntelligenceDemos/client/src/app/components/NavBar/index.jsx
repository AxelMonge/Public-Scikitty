import Link from 'next/link'

const NavBar = ({ }) => {
    return (
        <nav className="bg-nav-color h-10 flex items-center text-white sticky top-0">
            <ul className='flex justify-evenly w-full'>
                <li className='hover:text-gray-200'>
                    <Link href={"/"}> Tools </Link>
                </li>
                <li className='hover:text-gray-200'>
                    <Link href={"/AboutUs"}> About Us </Link>
                </li>
            </ul>
        </nav>
    )
}

export default NavBar;